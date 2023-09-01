import multiprocessing as mp
import argparse
import json
from longchat.retrievers import *
from longchat.retrievers.build_json_index import JSONLReader

# model = "text-davinci-003"
model = "gpt-3.5-turbo"
# model = "gpt-4"
OPENAI_API_KEY = ""

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluating the LLM for API calls possibly with retriever."
    )
    parser.add_argument(
        "--use_retriever",
        default=False,
        help="whether to use retriever or not",
    )
    parser.add_argument(
        "--retriever",
        default="llama_index",
        help="which retriever to use in our dataset",
    )
    parser.add_argument(
        "--api_dataset_path",
        default="",
        help="which dataset to use",
    )
    parser.add_argument(
        "--vicuna_dataset_path",
        default="",
        help="which dataset to use",
    )
    parser.add_argument(
        "--num_doc",
        default=5,
        help="number of docs to use",
    )
    args = parser.parse_args()
    return args

args = parse_args()
if args.use_retriever:
    if args.retriever == "llama_index":
        if os.path.exists(args.retriever + '_dataset_index.json'):
            print('data index already saved')
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
            from llama_index import GPTSimpleVectorIndex
            index = GPTSimpleVectorIndex.load_from_disk(args.retriever + 'dataset_index.json')
        else:
            print('data index being created')
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
            documents = JSONLReader().load_data(args.api_dataset_path)
            from llama_index import GPTSimpleVectorIndex
            index = GPTSimpleVectorIndex.from_documents(documents)
            index.save_to_disk(args.retriever + 'dataset_index.json')
        retriever = LlamaIndexRetriever(index=index)
    elif args.retriever == "bm25":
        from rank_bm25 import BM25Okapi
        corpus = []
        with open(args.api_dataset_path, 'r') as f:
            for line in f:
                corpus.append(json.loads(line))
        tokenized_corpus = [str(doc).split(" ") for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        retriever = BM25Retriever(index=bm25, corpus=corpus)
    else:
        assert False

with open(args.vicuna_dataset_path, 'r') as f:
    vicuna_data = json.load(f)

def add_retrieval(input):
    conversation = input['conversations']
    assert len(conversation) == 2
    query = conversation[0]['value']
    retrieved_doc = retriever.get_relevant_documents(query)

    for i in range(args.num_doc):
        content = retrieved_doc[i].page_content
        query = query + "\nRelated Doc " + str(i+1) + ": " + content
    
    return {"id": input["id"], "conversations": [{"from": "human", "value": query}, conversation[1]]}


with mp.Pool(16) as pool:
    outputs = pool.map(add_retrieval, [for input in vicuna_data])
    pool.close()
    pool.join()

# Write the list of JSONs to a file
with open('hf.json', 'w') as outfile:
    json.dump(outputs, outfile, indent=2)
