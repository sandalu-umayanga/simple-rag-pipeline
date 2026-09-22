import os
import time
import warnings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()


def load_documents(docs_path="docs"):
    print(f"Loading documents from {docs_path}...")
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Directory '{docs_path}' does not exist.")

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    )
    documents = loader.load()

    if len(documents) == 0:
        raise ValueError(f"No text files found in directory '{docs_path}'.")

    print(f"Loaded {len(documents)} documents.")
    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=150):
    print(f"Splitting documents into chunks (size={chunk_size}, overlap={chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db", batch_size=50):
    print(f"Initializing vector store in {persist_directory} with NVIDIA Embeddings...")

    # Initialize NVIDIA Embeddings
    embedding_model = NVIDIAEmbeddings(
        model="nvidia/nemotron-3-embed-1b",
        truncate="END"  # Truncates inputs safely if they exceed max token length
    )

    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )

    total_chunks = len(chunks)
    print(f"Embedding {total_chunks} chunks in batches of {batch_size}...")

    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        print(f"-> Ingesting chunks {i + 1} to {min(i + batch_size, total_chunks)} of {total_chunks}...")
        vector_store.add_documents(batch)
        time.sleep(0.5)

    print(f"\nVector store successfully populated at '{persist_directory}'.")
    return vector_store


def main():
    # Remove any old Chroma vector store to prevent dimension mismatch issues
    documents = load_documents()
    chunks = split_documents(documents)
    create_vector_store(chunks)


if __name__ == "__main__":
    main()