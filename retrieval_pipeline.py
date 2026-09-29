from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

load_dotenv()
persist_directory = "db/chroma_db"

# 1. Load embeddings and vector store
embedding_model = NVIDIAEmbeddings(
    model="nvidia/nemotron-3-embed-1b", truncate="END"
)

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.3,
    },
)

# 2. Initialize chat model with a reliable NIM endpoint
llm = ChatNVIDIA(
    model="meta/llama3-8b-instruct",
    temperature=0.1,
    max_tokens=512,
)

# 3. Prompt template
template = """You are a helpful assistant. Use ONLY the following retrieved context to answer the user's question.
If the answer cannot be found in the context, respond with "I cannot answer this question based on the provided documents."

Context:
{context}

Question:
{question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)


# 4. Context formatter
def format_docs(docs):
  return "\n\n".join(
      f"--- Document {i} ---\n{doc.page_content}"
      for i, doc in enumerate(docs, 1)
  )


# 5. Build RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. Execute query
query = "Which island does SpaceX lease for its launches in the pacific"

print(f"User Query: {query}\n")

# Run retrieval & generation
answer = rag_chain.invoke(query)
print(f"LLM answer:\n{answer}")