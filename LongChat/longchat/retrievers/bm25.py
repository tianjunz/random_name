from typing import Any, Dict, List, cast

from pydantic import BaseModel, Field

from longchat.retrievers.schema import BaseRetriever, Document


class BM25Retriever(BaseRetriever, BaseModel):
    """Question-answering with sources over an LlamaIndex data structure."""

    index: Any
    corpus: Any
    query_kwargs: Dict = Field(default_factory=dict)

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant for a query."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "You need to install `pip install llama-index` to use this retriever."
            )

        tokenized_query = query.split(" ")
        bm25_docs = self.index.get_top_n(tokenized_query, self.corpus, n=1)
        docs = []
        for source_node in bm25_docs:
            metadata = {}
            docs.append(
                Document(page_content=str(source_node), metadata=metadata)
            )
        return docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError("LlamaIndexRetriever does not support async")

