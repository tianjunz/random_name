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
        self.loss = 'gumbel'

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
        # retrieve_docs: Optional[list] = None,
        # input_queries: Optional[list] = None,
    ) -> Union[Tuple, transformers.modeling_outputs.BaseModelOutputWithPast]:
        # Retrieve information
        question_enc_outputs = self.question_encoder(
            input_ids=question_input_ids, attention_mask=question_attention_mask, return_dict=True
        )
        question_encoder_last_hidden_state = question_enc_outputs[0].mean(dim=1)  # hidden states of question encoder

        question_embeds = question_encoder_last_hidden_state.unsqueeze(1)
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
        docs_embeds = self.question_encoder(
            input_ids=docs_input_ids.reshape(batch_size*num_input_docs, -1), attention_mask=docs_attention_mask.reshape(batch_size*num_input_docs, -1), return_dict=True
        )[0].mean(dim=1)
        docs_embeds = docs_embeds.reshape(batch_size, num_input_docs, -1)

        # Normalize embeddings
        question_embeds_norm = question_embeds / torch.norm(question_embeds, dim=-1, keepdim=True)
        docs_embeds_norm = docs_embeds / torch.norm(docs_embeds, dim=-1, keepdim=True)
        logits = torch.sum(question_embeds * docs_embeds, dim=-1) / math.sqrt(question_embeds.shape[-1])
        probs = torch.nn.functional.softmax(logits, dim=-1)
        # print(torch.max(probs, dim=-1), torch.min(probs, dim=-1))

        # Gumbel Softmax
        if self.loss == 'gumbel':
            gumbel_logits = torch.nn.functional.gumbel_softmax(logits, hard=True)
            doc_indices = torch.nonzero(gumbel_logits)
        else:
            doc_indices = torch.distributions.categorical.Categorical(logits=logits).sample()
            doc_indices = torch.cat([torch.arange(logits.shape[0], dtype=doc_indices.dtype).to(doc_indices.device).unsqueeze(1), doc_indices.unsqueeze(1)], dim=-1)
        input_ids = []
        attention_mask = []
        if labels is not None:
            final_labels = [] 
        input_probs = []
        for doc_index in doc_indices:
            input_ids.append(retrieve_input_ids[doc_index[0], doc_index[1]].unsqueeze(0))
            attention_mask.append(retrieve_attention_mask[doc_index[0], doc_index[1]].unsqueeze(0))
            if labels is not None:
                final_labels.append(labels[doc_index[0], doc_index[1]].unsqueeze(0))
            # input_probs.append(probs[doc_index[0], doc_index[1]].unsqueeze(0))
            if self.loss == 'gumbel':
                # nll_probs = - gumbel_logits + torch.logsumexp(gumbel_logits, dim=-1)
                input_probs.append(gumbel_logits[doc_index[0], doc_index[1]].unsqueeze(0))
            else:
                nll_probs = - logits + torch.logsumexp(logits, dim=-1)
                input_probs.append(nll_probs[doc_index[0], doc_index[1]].unsqueeze(0))

        input_ids = torch.cat(input_ids, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)
        if labels is not None:
            final_labels = torch.cat(final_labels, dim=0)
            labels = final_labels
        input_probs = torch.cat(input_probs, dim=0)
        '''
        n_retrieve_docs = []
        for doc_index in doc_indices:
            n_retrieve_docs.append(retrieve_docs[doc_index[0]][doc_index[1]])
        

        # compute 
        retrieve_input_queries = []
        for input_query, n_retrieve_doc in zip(input_queries, n_retrieve_docs):
            query = "Question: " + input_query
            for i in range(n_docs):
                query = query + "\nRelated Doc " + str(i+1) + ": " + n_retrieve_doc[i]
            retrieve_input_queries.append(query)
        
        inputs = tokenizer(
            retrieve_input_queries,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        '''


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
            if self.loss == 'gumbel': 
                loss = loss_fct(shift_logits, shift_labels)
            else:
                shift_logits = shift_logits + input_probs
                loss = loss_fct(shift_logits, shift_labels)

            IGNORE_INDEX = LabelSmoother.ignore_index
            loss = loss.sum(-1)/ (shift_labels.ne(IGNORE_INDEX).sum(-1) + 1e-6)
            
            if self.loss == 'gumbel':
                # drop = (loss > 2).float()
                # loss = loss.detach() * drop + loss * (1 - drop)
                loss = (loss * input_probs).mean()
            else:
                loss = loss.mean()

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
