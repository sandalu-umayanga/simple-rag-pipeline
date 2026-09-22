import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


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

def main():
    # Load documents
    documents = load_documents()








if __name__ == "__main__":
    main()