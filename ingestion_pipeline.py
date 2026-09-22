# Import required libraries
import os
import warnings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Suppress deprecation warnings and load environment variables
warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()


def load_documents(docs_path="docs"):
    """Load text documents from a specified directory."""
    print(f"Loading documents from {docs_path}...")
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Directory '{docs_path}' does not exist.")

    # Create a directory loader to load all .txt files
    loader = DirectoryLoader(path=docs_path, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()

    # Ensure documents were found
    if len(documents) == 0:
        raise ValueError(f"No text files found in directory '{docs_path}'.")

    print(f"Loaded {len(documents)} documents.")
    return documents


def split_documents(documents, chunk_size=800, chunk_overlap=100):
    """Split documents into smaller chunks for embedding."""
    print(f"Splitting documents into chunks of size {chunk_size}...")
    # Create a recursive text splitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist a vector store using Google's embedding model."""
    print(f"Creating vector store in {persist_directory} using Gemini Embeddings...")

    # Initialize Google's embedding model
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-001"
    )

    # Create vector store from documents with cosine similarity metric
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print(f"Vector store created and saved to {persist_directory}.")
    return vector_store


def main():
    """Main function to orchestrate the ingestion pipeline."""
    # Load documents from disk
    documents = load_documents()
    # Split documents into chunks
    chunks = split_documents(documents)
    # Create and persist vector store
    create_vector_store(chunks)


if __name__ == "__main__":
    main()