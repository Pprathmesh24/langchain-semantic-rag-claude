import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyMuPDFLoader, Docx2txtLoader, CSVLoader, UnstructuredHTMLLoader
# pyrefly: ignore [missing-import]
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from config import EMBEDDING_MODEL, CHROMA_DIR, RAG_DATA_DIR

load_dotenv()

# Maps each supported file extension to its loader class and any kwargs it needs.
# To support a new file type, just add a new entry here — no other code changes needed.
FILE_LOADERS = {
    ".txt":  (TextLoader,            {"encoding": "utf-8"}),
    ".pdf":  (PyMuPDFLoader,         {}),
    ".docx": (Docx2txtLoader,        {}),
    ".csv":  (CSVLoader,             {}),
    ".html": (UnstructuredHTMLLoader, {}),
}


def load_documents(directory: str = RAG_DATA_DIR) -> list:
    documents = []

    # Run a separate DirectoryLoader for each file type so each uses its correct loader.
    # A single DirectoryLoader can only use one loader class, hence the loop.
    for ext, (loader_cls, loader_kwargs) in FILE_LOADERS.items():
        loader = DirectoryLoader(
            directory,
            glob=f"**/*{ext}",
            loader_cls=loader_cls,
            loader_kwargs=loader_kwargs,
            show_progress=True,
            silent_errors=True  # skip files that fail to parse instead of crashing
        )
        docs = loader.load()
        if docs:
            print(f"Loaded {len(docs)} {ext} file(s)")
        documents.extend(docs)

    if len(documents) == 0:
        raise FileNotFoundError(f"No supported files found in '{directory}'. Supported types: {list(FILE_LOADERS.keys())}")

    print(f"\nTotal documents loaded: {len(documents)}")
    return documents



def chunk_documents(documents: list, embeddings: HuggingFaceEmbeddings) -> list:
    '''SemanticChunker embeds each sentence and splits when it detects a spike in
        distance between consecutive sentences — meaning the topic has shifted.
        "percentile" cuts at the 95th percentile of all distances by default,
        so only the biggest topic shifts become boundaries.'''
    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile"
    )
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks


def load_embedding_model(model_name: str = EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    print(f"Loading embedding model: {model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    print("Embedding model loaded successfully")
    return embeddings


def embed_and_store(chunks: list, embeddings: HuggingFaceEmbeddings, persist_directory: str = CHROMA_DIR) -> Chroma:
    print(f"Storing {len(chunks)} chunks in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print(f"Successfully stored {vectorstore._collection.count()} chunks in ChromaDB at '{persist_directory}'")
    return vectorstore


def main():
    print("Main Function")

    #1. Load the files
    documents = load_documents("rag_data")

    for i, doc in enumerate(documents):
        print(f"\n--- Document {i + 1} ---")
        print(f"Metadata : {doc.metadata}")
        print(f"Content  : {doc.page_content[:200]}")

    #2. Load embedding model first — needed for both chunking and storing
    embedding_model = load_embedding_model()

    #3. Chunk using semantic boundaries
    chunks = chunk_documents(documents, embedding_model)

    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- Chunk {i + 1} ---")
        print(f"Metadata : {chunk.metadata}")
        print(f"Content  : {chunk.page_content[:300]}")
        print(f"Length   : {len(chunk.page_content)} chars")

    #4. Store in Vector DB
    vectorstore = embed_and_store(chunks, embedding_model)
    print(f"\nPipeline complete. Vector store ready at 'chroma_db'")

if __name__ == "__main__":
    main()
