from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    BartForConditionalGeneration,
    BatchEncoding,
    RagConfig,
    RagModel,
    RagSequenceForGeneration,
    RagTokenForGeneration,
    RagTokenizer,
    RagRetriever,
    AutoModelForCausalLM,
    LlamaForCausalLM,
)
import transformers
import torch
from typing import Optional, List, Union, Tuple
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.trainer_pt_utils import LabelSmoother
import math
from fastchat.model.model_adapter import get_conversation_template

N_SAMPLES = 3
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

class RagForCausalLM(LlamaForCausalLM):
    def setup(
        self,
        question_encoder,
        question_tokenizer,
        # question_max_length,
        # max_length,
        tokenizer, 
        question_encoder_config,
    ):
        self.question_encoder = question_encoder 
        self.question_tokenizer = question_tokenizer
        # self.question_max_length = question_max_length
        # self.max_length = max_length
        self.tokenizer = tokenizer
        self.question_encoder_config = question_encoder_config
        # Added an gate for deciding whether to drop retriever or to keep it
        self.use_retriever = torch.nn.Linear(self.question_encoder_config.d_model, 2)
        self.loss = 'multinomial'

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        n_docs: Optional[int] = None,
        question_input_ids: torch.LongTensor = None,
        question_attention_mask: Optional[torch.Tensor] = None,
        docs_input_ids: Optional[torch.LongTensor] = None,
        docs_attention_mask: Optional[torch.Tensor] = None,
        retrieve_input_ids: Optional[torch.LongTensor] = None,
        retrieve_attention_mask: Optional[torch.Tensor] = None,
        retrieve_docs: Optional[torch.LongTensor] = None,
        input_queries: Optional[torch.LongTensor] = None,
        label_queries: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, transformers.modeling_outputs.BaseModelOutputWithPast]:
        # Retrieve information
        # question_enc_outputs = self.question_encoder(
        #     input_ids=question_input_ids, attention_mask=question_attention_mask, return_dict=True
        # )
        # question_encoder_last_hidden_state = question_enc_outputs[0].mean(dim=1)  # hidden states of question encoder

        # question_embeds = question_encoder_last_hidden_state.unsqueeze(1)
        '''
        docs = []
        for input_doc in retrieve_docs:
            docs += input_doc
            num_input_docs = len(input_doc)
        doc_encoding = self.question_tokenizer(
            docs,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        '''
        batch_size, num_input_docs, _ = docs_input_ids.shape
        logits = self.question_encoder(
            # input_ids=docs_input_ids.reshape(batch_size*num_input_docs, -1), attention_mask=docs_attention_mask.reshape(batch_size*num_input_docs, -1), return_dict=True
            input_ids=docs_input_ids.reshape(batch_size*num_input_docs, -1), attention_mask=docs_attention_mask.reshape(batch_size*num_input_docs, -1), return_dict=True
        ).logits
        logits = logits.reshape(batch_size, num_input_docs, 2)
        encoder_logits = logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        # docs_embeds = docs_embeds.reshape(batch_size, num_input_docs, -1)

        # Normalize embeddings
        '''
        question_embeds_norm = question_embeds / torch.norm(question_embeds, dim=-1, keepdim=True)
        docs_embeds_norm = docs_embeds / torch.norm(docs_embeds, dim=-1, keepdim=True)
        logits = torch.sum(question_embeds * docs_embeds, dim=-1) / math.sqrt(question_embeds.shape[-1])
        probs = torch.nn.functional.softmax(logits, dim=-1)
        # print(torch.max(probs, dim=-1), torch.min(probs, dim=-1))
        '''

        # Gumbel Softmax
        if self.loss == 'gumbel':
            gumbel_logits = torch.nn.functional.gumbel_softmax(logits, hard=True)
            doc_indices = torch.nonzero(gumbel_logits)
        elif self.loss == 'softmax':
            doc_indices = torch.distributions.categorical.Categorical(logits=logits).sample()
            doc_indices = torch.cat([torch.arange(logits.shape[0], dtype=doc_indices.dtype).to(doc_indices.device).unsqueeze(1), doc_indices.unsqueeze(1)], dim=-1)
        elif self.loss == 'multinomial': 
            # doc_indices = torch.multinomial(probs, N_SAMPLES, False)
            gumbel_logits = torch.nn.functional.gumbel_softmax(logits.detach(), hard=True)[:, :, 0]
            topk_indices = torch.nn.functional.one_hot(torch.topk(probs[:, :, 0], k=N_SAMPLES, dim=1)[1], num_classes=logits.shape[1])
            topk_indices = torch.sum(topk_indices, dim=1)
            gumbel_logits = gumbel_logits * topk_indices
            # eps = 1e-3
            # gumbel_probs = torch.clamp(probs[:, :, 0], min=eps, max=1-eps)
            gumbel_logits = (gumbel_logits - probs[:, :, 0]).detach() + probs[:, :, 0]
            doc_indices = torch.nonzero(gumbel_logits)
            print(len(doc_indices))
            assert len(doc_indices) <= N_SAMPLES
            # doc_indices = torch.cat([torch.arange(logits.shape[0], dtype=doc_indices.dtype).to(doc_indices.device).unsqueeze(1), doc_indices], dim=-1)

        # input_ids = []
        # attention_mask = []
        if labels is not None:
            final_labels = [] 
        input_probs = []
        for i in range(batch_size):
            # input_ids.append(retrieve_input_ids[doc_index[0], doc_index[1]].unsqueeze(0))
            # attention_mask.append(retrieve_attention_mask[doc_index[0], doc_index[1]].unsqueeze(0))
            # if labels is not None:
            #     final_labels.append(labels[doc_index[0], doc_index[1]].unsqueeze(0))
            # input_probs.append(probs[doc_index[0], doc_index[1]].unsqueeze(0))
            if self.loss == 'gumbel':
                # nll_probs = - gumbel_logits + torch.logsumexp(gumbel_logits, dim=-1)
                input_probs.append(gumbel_logits[i, doc_index[1]].unsqueeze(0))
            elif self.loss == 'softmax':
                nll_probs = - logits + torch.logsumexp(logits, dim=-1)
                input_probs.append(nll_probs[i, doc_index[1]].unsqueeze(0))
            elif self.loss == 'multinomial':
                '''
                _input_probs = []
                for _doc_index in doc_index:
                    _input_probs.append(probs[i, _doc_index].unsqueeze(0))
                _input_probs = torch.cat(_input_probs, dim=0).unsqueeze(0)
                input_probs.append(_input_probs)
                '''
                _input_probs = torch.ones([1]).to(gumbel_logits.device)
                for _doc_index in doc_indices:
                    _input_probs *= gumbel_logits[_doc_index[0], _doc_index[1]].unsqueeze(0)
                input_probs.append(_input_probs)
        input_probs = torch.cat(input_probs, dim=0)

        # compute 
        input_queries = self.tokenizer.batch_decode(input_queries, skip_special_tokens=True)
        label_queries = self.tokenizer.batch_decode(label_queries, skip_special_tokens=True)
        raw_docs = []
        for retrieve_doc in retrieve_docs:
            raw_doc = self.tokenizer.batch_decode(retrieve_doc, skip_special_tokens=True)
            raw_docs.append(raw_doc)
        retrieve_docs = raw_docs
        retrieve_input_queries = []
        for input_query, n_retrieve_doc in zip(input_queries, retrieve_docs):
            query = "Question: " + input_query
            # TODO: Random permute questions
            rand_indices = torch.randperm(len(doc_indices))
            doc_indices = doc_indices[rand_indices]
            for i in range(len(doc_indices)):
                query = query + "\nRelated Doc " + str(i+1) + ": " + n_retrieve_doc[doc_indices[i][1]]
            retrieve_input_queries.append(query)
  
        # Get the labels
        conv = get_conversation_template("vicuna")
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        # Apply prompt templates
        conversations = []
        for i, (source, target) in enumerate(zip(retrieve_input_queries, label_queries)):
            conv.messages = []
            role = roles["human"]
            conv.append_message(role, source)
            role = roles["gpt"]
            conv.append_message(role, target)
            conversations.append(conv.get_prompt())
        # Tokenize conversations
        input_ids = self.tokenizer(
            conversations,
            return_tensors="pt",
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        ).input_ids.to(input_ids.device)
        targets = input_ids.clone()

        # Mask targets
        sep = conv.sep + conv.roles[1] + ": "
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(self.tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_TOKEN_ID
            for i, rou in enumerate(rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                round_len = len(self.tokenizer(rou).input_ids)
                instruction_len = len(self.tokenizer(parts[0]).input_ids) - 2

                target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
                cur_len += round_len
            target[cur_len:] = IGNORE_TOKEN_ID

            if False:
                z = target.clone()
                z = torch.where(z == IGNORE_TOKEN_ID, self.tokenizer.unk_token_id, z)
                rank0_print(self.tokenizer.decode(z))

            if cur_len < self.tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_TOKEN_ID
                    rank0_print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

        input_ids = input_ids
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        labels = targets

        # output = self.generator(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=labels)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            if self.loss == 'multinomial':
                shift_logits = shift_logits + torch.log(input_probs).unsqueeze(-1)
            loss = loss_fct(shift_logits, shift_labels)

            IGNORE_INDEX = LabelSmoother.ignore_index
            loss = loss.sum(-1)/ (shift_labels.ne(IGNORE_INDEX).sum(-1) + 1e-6)
            
            if self.loss == 'multinomial':
                # loss = (loss * input_probs).mean()
                acc = torch.sum(torch.nn.functional.gumbel_softmax(encoder_logits, hard=True)[:, :, 0], dim=1) / (encoder_logits.shape[1])
                threshold = N_SAMPLES/encoder_logits.shape[1]
                # loss += 0.01 * torch.sum((acc - threshold)**2 * (acc > threshold)) / torch.sum(acc > threshold + 1e-6)
                # loss += 10 * torch.mean((acc - threshold)**2)
                # print(doc_indices, ' loss ', torch.mean((acc - threshold)**2))
            '''
            if self.loss == 'gumbel':
                # drop = (loss > 2).float()
                # loss = loss.detach() * drop + loss * (1 - drop)
                loss = (loss * input_probs).mean()
            else:
                loss = (loss * ((1 - input_probs.mean(-1)).detach() + input_probs.mean(-1))).mean()
            '''

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # def generate(
    #     input_ids: Optional[torch.LongTensor] = None,
    #     attention_mask: Optional[torch.LongTensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_values: Optional[List[torch.FloatTensor]] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     return_dict: Optional[bool] = None,
    #     n_docs: Optional[int] = None,
    #     question_input_ids: torch.LongTensor = None,
    #     question_attention_mask: Optional[torch.Tensor] = None,
    #     docs_input_ids: Optional[torch.LongTensor] = None,
    #     docs_attention_mask: Optional[torch.Tensor] = None,
    #     retrieve_input_ids: Optional[torch.LongTensor] = None,
    #     retrieve_attention_mask: Optional[torch.Tensor] = None,
    #     logits_processor: Optional[LogitsProcessorList] = LogitsProcessorList(),
    #     stopping_criteria: Optional[StoppingCriteriaList] = StoppingCriteriaList(),
    #     **kwargs,
    # )    

'''
    def prepare_inputs_for_generation(
        self, 
        input_ids, 
        past_key_values=None, 
        attention_mask=None, 
        inputs_embeds=None, 
        question_input_ids=None,
        question_attention_mask=None,
        docs_input_ids=None,
        docs_attention_mask=None,
        retrieve_input_ids=None,
        retrieve_attention_mask=None, 
        **kwargs
    ):

        if not past_key_values:
            # Retrieve information
            question_enc_outputs = self.question_encoder(
                input_ids=question_input_ids, attention_mask=question_attention_mask, return_dict=True
            )
            question_encoder_last_hidden_state = question_enc_outputs[0].mean(dim=1)  # hidden states of question encoder

            question_embeds = question_encoder_last_hidden_state.unsqueeze(1)
            batch_size, num_input_docs, _ = docs_input_ids.shape
            docs_embeds = self.question_encoder(
                input_ids=docs_input_ids.reshape(batch_size*num_input_docs, -1), attention_mask=docs_attention_mask.reshape(batch_size*num_input_docs, -1), return_dict=True
            )[0].mean(dim=1)
            docs_embeds = docs_embeds.reshape(batch_size, num_input_docs, -1)

            # Normalize embeddings
            question_embeds_norm = question_embeds / torch.norm(question_embeds, dim=-1, keepdim=True)
            docs_embeds_norm = docs_embeds / torch.norm(docs_embeds, dim=-1, keepdim=True)
            logits = torch.sum(question_embeds * docs_embeds_norm, dim=-1)
            indices = torch.argmax(logits, dim=-1)
            batch_arange = torch.arange(batch_size).to(indices.device)
            if inputs_embeds is not None and past_key_values is None:
                assert False
                new_inputs_embeds = retrieve_inputs_embeds[batch_arange, indices]
            else:
                new_input_ids = retrieve_input_ids[batch_arange, indices]
                new_attention_mask = retrieve_attention_mask[batch_arange, indices]

                new_position_ids = kwargs.get("position_ids", None)
                if new_attention_mask is not None and new_position_ids is None:
                    # create position_ids on the fly for batch generation
                    new_position_ids = new_attention_mask.long().cumsum(-1) - 1
                    new_position_ids.masked_fill_(new_attention_mask == 0, 1)
        else:
            new_input_ids = input_ids
            new_attention_mask = attention_mask

            position_ids = kwargs.get("position_ids", None)
            if attention_mask is not None and position_ids is None:
                # create position_ids on the fly for batch generation
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                if past_key_values:
                    position_ids = position_ids[:, -1].unsqueeze(-1)
            new_position_ids = position_ids

        if past_key_values:
            new_input_ids = new_input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            assert False
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": new_input_ids}

        model_inputs.update(
            {
                "position_ids": new_position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": new_attention_mask,
            }
        )
        return model_inputs
'''
