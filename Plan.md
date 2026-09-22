A complete Retrieval-Augmented Generation (RAG) system consists of two major workflows: the **Indexing Pipeline** (offline processing of your files) and the **Inference & Evaluation Pipeline** (online question answering and quality checks).

---

### Step 1: Recursive Document Ingestion

Traverse the root folder and all nested subfolders to locate and read every document (such as `.txt`, `.pdf`, `.md`, or `.docx`) into a structured representation.

* **Why it matters:** A RAG system cannot retrieve knowledge it cannot read. It must extract clean text while retaining lineage—such as source file path and relative hierarchy.
* **How it works:**
* The ingestion module executes a filesystem walk (e.g., using Python’s `os.walk` or `pathlib.Path.rglob("*")`).
* Dedicated parsers process specific file extensions: standard readers for plain text, specialized parsers like `pypdf` or `pymupdf` for PDFs, and `docx` parsers for Word files.
* The output is normalized into memory as a list of document objects containing raw content and a dictionary of metadata: `{"source_path": "folder/subfolder/file.pdf", "page": 2}`.


* **What happens if missed:** The pipeline fails to access nested documents or encounters unparsed binary data, crashing the loader or leaving significant blind spots in your knowledge base.

---

### Step 2: Document Chunking (Splitting)

Break down lengthy, multi-page texts into smaller, semantically coherent segments (chunks).

* **Why it matters:** LLMs and embedding models operate within finite context windows. Furthermore, large texts dilute semantic focus: an entire 50-page manual will have an average embedding representing everything generally, rather than pinpointing exact answers.
* **How it works:**
* A text splitter (e.g., `RecursiveCharacterTextSplitter`) segments the text using hierarchical separators (`\n\n`, `\n`, `. `, ` `) to avoid cutting mid-sentence.
* You define a **chunk size** (e.g., 500 to 1,000 characters or ~250–500 tokens) and an **overlap** (e.g., 50 to 100 characters).
* Overlap ensures that contextual transitions between boundaries are not clipped or lost.


* **What happens if missed:** You will exceed embedding model token limits, causing API errors. If you pass unbounded sections, relevant facts get lost in the noise of surrounding paragraphs.

---

### Step 3: Semantic Vector Embedding

Pass each chunk through an embedding model via an API (e.g., OpenAI `text-embedding-3-small` or Cohere) or a local transformer to produce a dense numerical vector.

* **Why it matters:** Computers cannot compute mathematical proximity on raw text strings. Embeddings represent the core conceptual meaning as coordinates in high-dimensional vector space (e.g., 1,536 dimensions).
* **How it works:**
* The chunk text is tokenized and fed to the neural network.
* The network outputs a normalized floating-point array.
* Similar semantic topics (e.g., *"revenue grew"* and *"sales increased"*) land close to each other in this coordinate space, even if they share zero identical words.


* **What happens if missed:** Vector search becomes impossible. You would be restricted to traditional keyword matching (like SQL `LIKE` or basic RegEx), which fails on synonyms, phrasing variations, and typos.

---

### Step 4: Vector Database Indexing & Storage

Store the generated vector embeddings alongside their original text snippets and metadata in a vector database (e.g., ChromaDB, LanceDB, Qdrant, or Pinecone).

* **Why it matters:** Searching through thousands of vectors via linear brute-force distance comparisons quickly degrades latency. A dedicated vector database indexes vectors for millisecond similarity lookups.
* **How it works:**
* The database creates an index using algorithms like HNSW (Hierarchical Navigable Small World) or IVF (Inverted File).
* Chunks are upserted with unique IDs, text payloads, vectors, and source metadata.


* **What happens if missed:** You must re-embed every document on every query, generating massive API costs and unacceptable multi-second delays.

---

### Step 5: Retrieval & Similarity Search

Convert the user's incoming question into a vector and search the database for the most relevant document chunks.

* **Why it matters:** This acts as the bridge connecting your external knowledge base to the prompt sent to the LLM. High-quality retrieval is the single most critical factor for accurate generation.
* **How it works:**
* The user's query is converted to a vector using the **exact same embedding model** used in Step 3.
* The vector database calculates cosine distance or dot-product similarity between the query vector and all stored document vectors.
* It returns the top-$K$ nearest neighbors (e.g., the top 3–5 most relevant text chunks) along with their metadata.


* **What happens if missed:** The LLM receives only the user query with no grounding material, forcing it to fall back on its base training weights and hallucinating facts about proprietary documents.

---

### Step 6: Augmented Prompt Assembly & LLM Generation

Inject the retrieved chunks into a structured system prompt and send it to your generative LLM API (e.g., GPT-4o, Claude 3.5 Sonnet, or Gemini).

* **Why it matters:** The LLM acts as the synthesis and reasoning engine that transforms disjointed raw text excerpts into a coherent, natural-language answer.
* **How it works:**
* Construct a prompt containing three key elements:
1. **System Guardrails:** Explicit instructions to answer *only* based on the context, cite sources, and admit ignorance if the text lacks sufficient data.
2. **Retrieved Context:** Chunks formatted clearly with labels (e.g., `[Source: folder/spec.pdf]: ...`).
3. **User Query:** The original question.


* Send the payload to the LLM API and stream back the response.


* **What happens if missed:** You are left with raw database fragments that the user must read and interpret manually without synthesis or direct answers.

---

### Step 7: System Evaluation & Quality Assurance

Run quantitative scoring on your RAG pipeline outputs using an evaluation framework (e.g., Ragas, TruLens, or an "LLM-as-a-judge" approach).

* **Why it matters:** Without systematic measurement, you cannot tell whether tuning chunk size, changing the embedding model, or modifying prompts improves accuracy or degrades performance.
* **How it works:**
* The pipeline is evaluated along the **RAG Triad**:
* **Context Relevance:** Did the retriever fetch chunks genuinely relevant to the query?
* **Groundedness / Faithfulness:** Is the LLM's generated response directly supported by the retrieved context, or does it introduce hallucinations?
* **Answer Relevance:** Does the answer directly address the specific question asked?


* Automated judges score each metric on a 0.0–1.0 scale to identify regressions.


* **What happens if missed:** "Silent failures" go unnoticed—retrieval noise, subtle hallucinations, and missing details will persist in production without visibility into which component failed.

---

### Standard Python Implementation Blueprint

```python
import os
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Ingestion: Walk directory & subdirectories
DOCS_DIR = "./my_documents"
loader = DirectoryLoader(
    DOCS_DIR, 
    glob="**/*", 
    loader_cls=TextLoader, 
    show_progress=True, 
    use_multithreading=True
)
raw_docs = loader.load()

# 2. Chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
chunks = text_splitter.split_documents(raw_docs)

# 3 & 4. Embedding & Vector Database
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory="./chroma_db"
)

# 5. Retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 4})

# 6. Response Generator
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

template = """Answer the question based only on the provided context. If you cannot find the answer, state that you do not know.

Context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(f"[Source: {d.metadata.get('source', 'Unknown')}]:\n{d.page_content}" for d in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Example run
response = rag_chain.invoke("What are the quarterly goals in our strategy doc?")
print(response)

```