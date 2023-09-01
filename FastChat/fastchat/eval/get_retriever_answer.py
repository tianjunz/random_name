import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import ray

from fastchat.model import get_conversation_template

from transformers import BertConfig, AutoTokenizer, AutoModelForSequenceClassification, T5Config, T5EncoderModel, T5Tokenizer, AutoConfig
from fastchat.train.retrievers import *
from fastchat.train.retrievers.build_json_index import JSONLReader
from fastchat.train.modeling_rag_backup import RagForCausalLM
from rank_bm25 import BM25Okapi

N_DOCS = 32

def run_eval(model_path, model_id, question_file, answer_file, num_gpus, question_model_path, api_file, retriever_type):
    # split question file into num_gpus files
    ques_jsons = []
    with open(os.path.expanduser(question_file), "r") as ques_file:
        for line in ques_file:
            ques_jsons.append(line)
   
    if retriever_type == "bm25":
        corpus = []
        with open(api_file, 'r') as f:
            for line in f:
                corpus.append(json.loads(line))
        tokenized_corpus = [str(doc).split(" ") for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        retriever = BM25Retriever(index=bm25, corpus=corpus) 
    elif retriever_type == "llama_index":
        OPENAI_API_KEY = ""
        if os.path.exists('../../'+api_file.split(".json")[0] + '_index.json'):
            print('data index already saved')
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
            from llama_index import GPTSimpleVectorIndex
            index = GPTSimpleVectorIndex.load_from_disk('../../'+api_file.split(".json")[0] + '_index.json')
        else:
            assert False
            print('data index being created')
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
            documents = JSONLReader().load_data(model_args.api_dataset_path)
            from llama_index import GPTSimpleVectorIndex
            index = GPTSimpleVectorIndex.from_documents(documents)
            index.save_to_disk(model_args.api_dataset_name.split(".json")[0] + '_index.json')
        retriever = LlamaIndexRetriever(index=index, query_kwargs={"similarity_top_k": 5})

    chunk_size = len(ques_jsons) // num_gpus
    ans_handles = []
    for i in range(0, len(ques_jsons), chunk_size):
        ans_handles.append(
            get_model_answers.remote(
                model_path, model_id, ques_jsons[i : i + chunk_size], question_model_path, retriever
            )
        )

    ans_jsons = []
    for ans_handle in ans_handles:
        ans_jsons.extend(ray.get(ans_handle))

    with open(os.path.expanduser(answer_file), "w") as ans_file:
        for line in ans_jsons:
            ans_file.write(json.dumps(line) + "\n")


@ray.remote(num_gpus=1)
@torch.inference_mode()
def get_model_answers(model_path, model_id, question_jsons, question_encoder_path, retriever):
    model_path = os.path.expanduser(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(
        model_path, low_cpu_mem_usage=True, torch_dtype=torch.float16
    ) 

    # Initialize encoder
    '''
    encoder = T5EncoderModel.from_pretrained(model_path,
        trust_remote_code=True,
    )
    '''
    encoder_dict = torch.load(model_path + '/pytorch_model-00003-of-00003.bin')
    # encoder = T5EncoderModel(T5Config())
    # encoder = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-yelp-polarity", num_labels=1, ignore_mismatched_sizes=True)
    encoder = AutoModelForSequenceClassification.from_pretrained("facebook/contriever", num_labels=2, ignore_mismatched_sizes=True)
    state_dict = {}
    for k, v in encoder_dict.items():
        if 'question_encoder.' in k or 'classifier' in k:
            state_dict[k.replace("question_encoder.", "")] = v
    encoder.load_state_dict(state_dict)

    encoder.config.use_cache = False
    encoder_tokenizer = AutoTokenizer.from_pretrained("facebook/contriever", num_labels=2, ignore_mismatched_sizes=True)
    '''
    encoder_tokenizer = T5Tokenizer.from_pretrained(
        # model_path,
        question_encoder_path,
        use_fast=False,
    )
    '''
    # encoder_tokenizer = T5Tokenizer(T5Config(), use_fast=False)
    # tokenizer.pad_token = tokenizer.unk_token
    '''
    model.setup(
        encoder,
        encoder_tokenizer,
        tokenizer,
        AutoConfig.from_pretrained(question_encoder_path),
    )
    '''
    encoder = encoder.cuda()
    model = model.cuda()

    ans_jsons = []
    ans_jsons2 = []
    for i, line in enumerate(tqdm(question_jsons)):
        ques_json = json.loads(line)
        idx = ques_json["question_id"]
        qs = ques_json["text"]
        # Added retriever
        # question = encoder_tokenizer([qs], padding="max_length", 
        #         max_length=encoder_tokenizer.model_max_length, truncation=True,).input_ids
        # retrieved_doc = retriever.get_relevant_documents(qs)
        # retrieve_docs = [doc.page_content for doc in retrieved_doc]
        retrieve_docs = ques_json["docs"]
        retrieve_docs = [qs + "\nRelated Doc " + str(1) + ": " + doc for doc in retrieve_docs]
        docs = encoder_tokenizer(retrieve_docs, padding="max_length", 
                max_length=encoder_tokenizer.model_max_length, truncation=True,).input_ids
        # docs_input_ids = docs["input_ids"].reshape(len(input_queries), N_DOCS, -1)
        # docs_attention_mask = docs["attention_mask"].reshape(len(input_queries), N_DOCS, -1)
        retrieve_input_queries = []
        for doc in retrieve_docs:
            query = "Question: " + qs
            query = query + "\nRelated Doc " + str(1) + ": " + doc
            retrieve_input_queries.append(query)

        prompts = []
        for retrieve_input_query in retrieve_input_queries:
            conv = get_conversation_template(model_id)
            conv.append_message(conv.roles[0], retrieve_input_query)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            prompts.append(prompt)
        input_ids = tokenizer(prompts).input_ids

        '''
        question_enc_outputs = encoder(
			input_ids=torch.as_tensor(question).cuda(), return_dict=True
		)
        question_encoder_last_hidden_state = question_enc_outputs[0].mean(dim=1)  # hidden states of question encoder
        question_embeds = question_encoder_last_hidden_state.unsqueeze(1)
        batch_size, num_input_docs = 1, N_DOCS
        docs_embeds = encoder(
            input_ids=torch.as_tensor(docs).cuda().reshape(batch_size*num_input_docs, -1), return_dict=True
        )[0].mean(dim=1)
        docs_embeds = docs_embeds.reshape(batch_size, num_input_docs, -1)
        # Normalize embeddings
        question_embeds_norm = question_embeds / torch.norm(question_embeds, dim=-1, keepdim=True)
        docs_embeds_norm = docs_embeds / torch.norm(docs_embeds, dim=-1, keepdim=True)
        logits = torch.sum(question_embeds * docs_embeds, dim=-1)
        '''
        batch_size, num_input_docs = 1, N_DOCS
        logits = encoder(
            input_ids=torch.as_tensor(docs).cuda().reshape(batch_size*num_input_docs, -1), return_dict=True
        ).logits
        logits = logits[:, 0].squeeze().unsqueeze(0)

        indices = torch.argmax(logits, dim=-1)
        batch_arange = torch.arange(batch_size).to(indices.device)
        input_ids = input_ids[indices[0]]
        outputs = ques_json["docs"][indices[0]]

        ans_id = shortuuid.uuid()
        ans_jsons.append(
            {
                "question_id": idx,
                "text": outputs,
                "answer_id": ans_id,
                "model_id": model_id,
                "metadata": {},
            }
        )
    return ans_jsons


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--question-encoder-model-path", type=str, default="t5-small")
    parser.add_argument("--api-file", type=str, required=True)
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answer-file", type=str, default="answer.jsonl")
    parser.add_argument("--retriever-type", type=str, default="bm25")
    parser.add_argument("--num-gpus", type=int, default=1)
    args = parser.parse_args()

    ray.init()
    run_eval(
        args.model_path,
        args.model_id,
        args.question_file,
        args.answer_file,
        args.num_gpus,
        args.question_encoder_model_path,
        args.api_file,
        args.retriever_type,
    )
