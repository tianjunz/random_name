# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import os
from dataclasses import dataclass, field
import json
import pathlib
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
from transformers import T5EncoderModel, T5Tokenizer, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template

from fastchat.train.retrievers import *
from fastchat.train.retrievers.build_json_index import JSONLReader
from fastchat.train.modeling_rag1 import RagForCausalLM
from rank_bm25 import BM25Okapi


IGNORE_TOKEN_ID = LabelSmoother.ignore_index
N_DOCS = 32

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    question_encoder_name_or_path: Optional[str] = field(default="t5-small")
    api_dataset_path: Optional[str] = field(default="playground/data/openai_hf-2023May20.json")
    api_dataset_name: Optional[str] = field(default="openai_hf-2023May20.json")
    retriever_type: Optional[str] = field(default="bm25")
    retriever_n_docs: Optional[int] = field(default=1)

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    question_encoder_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def preprocess(
    sources,
    docs, 
    tokenizer: transformers.PreTrainedTokenizer,
    encoder_tokenizer: transformers.PreTrainedTokenizer,
    retriever, 
    n_docs,
) -> Dict:
    conv = get_conversation_template("vicuna")
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    input_queries = []
    label_queries = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            if j == 0:
                input_queries.append(sentence["value"])
            if j == 1:
                label_queries.append(sentence["value"])

    # Get inputs
    query = encoder_tokenizer(
        input_queries,
        return_tensors="pt",
        padding="max_length",
        max_length=encoder_tokenizer.model_max_length,
        truncation=True,
    )
    question_input_ids = query["input_ids"]
    question_attention_mask = query["attention_mask"]
    retrieve_docs = []
    retrieve_docs_2d = []
    encoder_retrieve_docs = []
    '''
    for query in input_queries:
        retrieved_doc = retriever.get_relevant_documents(query)
        retrieve_docs += [doc.page_content for doc in retrieved_doc]
        retrieve_docs_2d.append([doc.page_content for doc in retrieved_doc])
    '''
    for input_query, doc in zip(input_queries, docs):
        retrieve_docs += [_doc for _doc in doc[:N_DOCS]]
        encoder_retrieve_docs += [input_query + "\nRelated Doc " + str(1) + ": " + _doc for _doc in doc[:N_DOCS]]
        retrieve_docs_2d.append(doc[:N_DOCS])

    docs = encoder_tokenizer(
        encoder_retrieve_docs,
        return_tensors="pt",
        padding="max_length",
        max_length=encoder_tokenizer.model_max_length,
        truncation=True,
    )
    docs_input_ids = docs["input_ids"].reshape(len(input_queries), N_DOCS, -1)
    docs_attention_mask = docs["attention_mask"].reshape(len(input_queries), N_DOCS, -1)

    retrieve_input_queries = []
    retrieve_input_queries_2d = []
    for input_query, retrieve_doc in zip(input_queries, retrieve_docs_2d):
        # temp_retrieve_input_queries = [input_query]
        temp_retrieve_input_queries = []
        # retrieve_input_queries.append(input_query)
        for doc in retrieve_doc:
            query = input_query
            query = query + "\nRelated Doc " + str(1) + ": " + doc
            retrieve_input_queries.append(query)
            temp_retrieve_input_queries.append(query)
        retrieve_input_queries_2d.append(temp_retrieve_input_queries)

    '''
    retrieve_inputs = tokenizer(
        retrieve_input_queries,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )
    n_input_docs = N_DOCS # +1 for no retrieval
    retrieve_input_ids = retrieve_inputs["input_ids"].reshape(len(input_queries), n_input_docs, -1)
    retrieve_attention_mask = retrieve_inputs["attention_mask"].reshape(len(input_queries), n_input_docs, -1)
    '''

    # Apply prompt templates
    conversations = []
    for i, (source, retrieve_queries) in enumerate(zip(sources, retrieve_input_queries_2d)):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        for retrieve_query in retrieve_queries:
            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                if sentence["from"] == "human": 
                    conv.append_message(role, retrieve_query)
                else:
                    conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

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
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID

            cur_len += round_len
        target[cur_len:] = IGNORE_TOKEN_ID

        if False:
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(tokenizer.decode(z))

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                rank0_print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    n_input_docs = N_DOCS
    input_ids = input_ids.reshape(len(input_queries), n_input_docs, -1)
    targets = targets.reshape(len(input_queries), n_input_docs, -1)

    input_queries = tokenizer(
        input_queries,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    label_queries = tokenizer(
        label_queries,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    retrieve_docs = tokenizer(
        retrieve_docs,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
        
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
        input_queries=input_queries,
        label_queries=label_queries,
        retrieve_docs=retrieve_docs,
        question_input_ids=question_input_ids,
        question_attention_mask=question_attention_mask,
        docs_input_ids=docs_input_ids,
        docs_attention_mask=docs_attention_mask,
        retrieve_input_ids=input_ids, 
        retrieve_attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
        raw_data, 
        tokenizer: transformers.PreTrainedTokenizer,
        encoder_tokenizer: transformers.PreTrainedTokenizer, 
        retriever,
        n_docs,
    ):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        docs = [example["docs"] for example in raw_data]
        data_dict = preprocess(sources, docs, tokenizer, encoder_tokenizer, retriever, n_docs)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]
        self.question_input_ids=data_dict["question_input_ids"]
        self.question_attention_mask=data_dict["question_attention_mask"]
        self.docs_input_ids=data_dict["docs_input_ids"]
        self.docs_attention_mask=data_dict["docs_attention_mask"]
        self.retrieve_input_ids=data_dict["retrieve_input_ids"]
        self.retrieve_attention_mask=data_dict["retrieve_attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
            question_input_ids=self.question_input_ids[i],
            question_attention_mask=self.question_attention_mask[i],
            docs_input_ids=self.docs_input_ids[i],
            docs_attention_mask=self.docs_attention_mask[i],
            retrieve_input_ids=self.retrieve_input_ids[i],
            retrieve_attention_mask=self.retrieve_attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, 
        tokenizer: transformers.PreTrainedTokenizer,
        encoder_tokenizer: transformers.PreTrainedTokenizer, 
        retriever,
        n_docs,
    ):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.encoder_tokenizer = encoder_tokenizer
        self.retriever = retriever
        self.n_docs = n_docs

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], [self.raw_data[i]["docs"]], self.tokenizer, 
            self.encoder_tokenizer, self.retriever, self.n_docs)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
            question_input_ids=ret["question_input_ids"][0],
            question_attention_mask=ret["question_attention_mask"][0],
            docs_input_ids=ret["docs_input_ids"][0],
            docs_attention_mask=ret["docs_attention_mask"][0],
            retrieve_input_ids=ret["retrieve_input_ids"][0],
            retrieve_attention_mask=ret["retrieve_attention_mask"][0],
            input_queries=ret["input_queries"][0],
            label_queries=ret["label_queries"][0],
            retrieve_docs=ret["retrieve_docs"],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, 
    encoder_tokenizer: transformers.PreTrainedTokenizer, 
    retriever,
    n_docs,
    data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")
    raw_data = json.load(open(data_args.data_path, "r"))

    # Split train/test
    perm = np.random.permutation(len(raw_data))
    split = int(len(perm) * 0.98)
    train_indices = perm[:split]
    eval_indices = perm[split:]
    train_raw_data = [raw_data[i] for i in train_indices]
    eval_raw_data = [raw_data[i] for i in eval_indices]
    rank0_print(f"#train {len(train_raw_data)}, #eval {len(eval_raw_data)}")

    train_dataset = dataset_cls(train_raw_data, tokenizer=tokenizer, encoder_tokenizer=encoder_tokenizer, retriever=retriever, n_docs=n_docs)
    eval_dataset = dataset_cls(eval_raw_data, tokenizer=tokenizer, encoder_tokenizer=encoder_tokenizer, retriever=retriever, n_docs=n_docs)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train_rag():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    # Initialize model
    # generator = transformers.AutoModelForCausalLM.from_pretrained(
    model = RagForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
    )
    model.config.use_cache = False
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    # Initialize encoder
    # encoder_tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-yelp-polarity")
    # encoder = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-yelp-polarity", num_labels=2, ignore_mismatched_sizes=True)
    encoder_tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
    encoder = AutoModelForSequenceClassification.from_pretrained('facebook/contriever', num_labels=2)
    encoder.config.use_cache = False
    # tokenizer.pad_token = tokenizer.unk_token
    model.setup(
        encoder,
        encoder_tokenizer,
        # training_args.question_encoder_max_length,
        # training_args.model_max_length,
        tokenizer, 
        AutoConfig.from_pretrained(model_args.question_encoder_name_or_path),
    )

    if model_args.retriever_type == "bm25":
        corpus = []
        with open(model_args.api_dataset_path, 'r') as f:
            for line in f:
                corpus.append(json.loads(line))
        tokenized_corpus = [str(doc).split(" ") for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        retriever = BM25Retriever(index=bm25, corpus=corpus)
    elif model_args.retriever_type == "llama_index": 
        OPENAI_API_KEY = ""
        if os.path.exists(model_args.api_dataset_name.split(".json")[0] + '_index.json'):
            print('data index already saved')
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
            from llama_index import GPTSimpleVectorIndex
            index = GPTSimpleVectorIndex.load_from_disk(model_args.api_dataset_name.split(".json")[0] + '_index.json')
        else:
            print('data index being created')
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
            documents = JSONLReader().load_data(model_args.api_dataset_path)
            from llama_index import GPTSimpleVectorIndex
            index = GPTSimpleVectorIndex.from_documents(documents)
            index.save_to_disk(model_args.api_dataset_name.split(".json")[0] + '_index.json')
        retriever = LlamaIndexRetriever(index=index, query_kwargs={"similarity_top_k": 5})

    data_module = make_supervised_data_module(tokenizer=tokenizer, encoder_tokenizer=encoder_tokenizer, retriever=retriever, n_docs=model_args.retriever_n_docs, data_args=data_args)
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train_rag()
