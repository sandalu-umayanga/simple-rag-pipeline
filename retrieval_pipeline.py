from langchain_chroma import Chroma
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from dotenv import load_dotenv


load_dotenv()
persist_directory = "db/chroma_db"

# Load embeddings and vector store
embedding_model = NVIDIAEmbeddings(
    model="nvidia/nemotron-3-embed-1b",
    truncate="END"
)

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
)

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k":5,
        "score_threshold":0.3  # only return chunks with cosine similarity >= 0.3
    }
)



# ====================================================================
# main working sequence
query = "Which island does SpaceX lease for its launches in the pacific"

relavent_docs = retriever.invoke(query)

print(f"User Query: {query}")

print("---Context---")

for i, doc in enumerate(relavent_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")