# Retrieval pipeline: loads the persisted vector store from disk and provides
# a two-step retriever:
#   Step 1 — MMR search via LangChain's built-in max_marginal_relevance_search_with_score
#   Step 2 — post-filter results below the similarity score threshold

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda
# pyrefly: ignore [missing-import]
from config import CHROMA_DIR, EMBEDDING_MODEL, SIMILARITY_THRESHOLD, MMR_FETCH_K, MMR_LAMBDA


def load_vectorstore(persist_directory: str = CHROMA_DIR, model_name: str = EMBEDDING_MODEL) -> Chroma:
    # We must use the same embedding model used during ingestion — Chroma stores
    # raw vectors, so a different model would produce incompatible query vectors.
    print(f"Loading vector store from '{persist_directory}'...")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # Chroma() connects to an existing persisted collection (unlike
    # Chroma.from_documents(), which creates a new one from scratch).
    # cosine similarity is specified here to match how embeddings were stored.
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print(f"Vector store loaded. Total chunks: {vectorstore._collection.count()}")
    return vectorstore


def get_retriever(
    vectorstore: Chroma,
    top_k: int = 5,
    score_threshold: float = SIMILARITY_THRESHOLD,
    fetch_k: int = MMR_FETCH_K,
    lambda_mult: float = MMR_LAMBDA
) -> RunnableLambda:

    def retrieve(query: str) -> list:
        # Step 1: LangChain's built-in MMR — fetches fetch_k candidates, selects
        # diverse top_k, returns (doc, distance) pairs.
        # Distance is cosine distance: 0 = identical, 1 = unrelated.
        results = vectorstore.max_marginal_relevance_search_with_score(
            query, k=top_k, fetch_k=fetch_k, lambda_mult=lambda_mult
        )

        # Step 2: convert cosine distance → relevance score (1 - distance),
        # then drop anything below the threshold.
        return [doc for doc, distance in results if (1 - distance) >= score_threshold]

    return RunnableLambda(retrieve)


def main():
    vectorstore = load_vectorstore()
    retriever = get_retriever(vectorstore)

    query = "What is the confidentiality clause?"
    print(f"\nQuery: {query}")
    results = retriever.invoke(query)

    if not results:
        print("No results above the similarity threshold.")
    else:
        for i, doc in enumerate(results):
            print(f"\n--- Result {i + 1} ---")
            print(f"Source : {doc.metadata['source']}")
            print(f"Content: {doc.page_content[:300]}")


if __name__ == "__main__":
    main()
