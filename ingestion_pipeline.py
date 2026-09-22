import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# loading documents from the "docs" directory as langchain Document objects
def load_documents(docs_path="docs"):
    # load text files from the example docs directory
    print(f"Loading documents from {docs_path}...")

    # check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Directory '{docs_path}' does not exist.")

    # load all text files in the directory
    loader = DirectoryLoader(
        path = docs_path,
        glob = "*.txt",
        loader_cls=TextLoader
    )

    documents = loader.load()

    if len(documents) == 0:
        raise ValueError(f"No text files found in directory '{docs_path}'.")

    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}: ")
        print(f" Source: {doc.metadata['source']}")
        print(f" Content length: {len(doc.page_content)} characters")
        print(f" Content preview: {doc.page_content[:200]}...")  # print first 200 characters
        print(f" Metadata: {doc.metadata}")

    
    print(f"Loaded {len(documents)} documents.")
    return documents


# split documents into chunks of specified size with optional overlap
def split_documents(documents, chunk_size=800, chunk_overlap=0): # chunk_size means number of characters in each chunk
    # split documents into chunks
    print(f"Splitting documents into chunks of size {chunk_size} with overlap {chunk_overlap}...")


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    for i, chunk in enumerate(chunks[:2]):
        print(f"\nChunk {i+1}: ")
        print(f" Content length: {len(chunk.page_content)} characters")
        print(f" Content preview: {chunk.page_content[:200]}...")  # print first 200 characters
        print(f" Metadata: {chunk.metadata}")

    print(f"Split into {len(chunks)} chunks.")
    return chunks



def main():
    # Load documents
    documents = load_documents()

    # Split documents into chunks
    chunks = split_documents(documents)








if __name__ == "__main__":
    main()