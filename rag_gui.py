import os
import json
import gradio as gr
import ollama

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# === Offline Environment Setup ===
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# === Configuration ===
DOC_DIR = "docs"
HISTORY_FILE = "memory/chat_history.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "qwen3:1.7b"

# === Helper Functions ===
def load_docs(directory):
    all_docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            path = os.path.join(directory, filename)
            loader = TextLoader(path, encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filename
            all_docs.extend(docs)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(all_docs)

def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.from_documents(chunks, embeddings)

def load_memory():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                data = f.read().strip()
                if not data:
                    return []
                return json.loads(data)
        except json.JSONDecodeError:
            print("⚠️ Warning: chat_history.json is corrupted. Starting fresh.")
            return []
    return []

def save_memory(memory):
    os.makedirs("memory", exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

def build_prompt(query, history, docs):
    past = "\n".join([f"User: {m['user']}\nAssistant: {m['assistant']}" for m in history[-5:]])
    context = "\n".join(
        f"({doc.metadata.get('source', 'unknown')}) {doc.page_content.strip()}" for doc in docs
    )
    return f"""
You are a helpful assistant. Use only the information in the CONTEXT and MEMORY below to answer the QUESTION.

- If the answer is clearly stated in the context, give a direct, concise response.
- If the answer is not in the context, say: \"I don't know based on the context provided.\"
- Do not guess or assume.

=== CONTEXT ===
{context}

=== MEMORY ===
{past}

=== QUESTION ===
{query}

=== ANSWER ===
"""

# === Load RAG System ===
docs = load_docs(DOC_DIR)
db = get_vectorstore(docs)
memory = load_memory()

def rag_chat(query):
    retrieved = db.similarity_search(query, k=3)

    context_preview = "\n".join(
        f"({doc.metadata.get('source', 'unknown')}) {doc.page_content.strip()[:100]}..."
        for doc in retrieved
    )

    prompt = build_prompt(query, memory, retrieved) if retrieved else build_prompt(query, memory, [])

    response = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
    answer = response["message"]["content"]

    memory.append({"user": query, "assistant": answer})
    save_memory(memory)

    return context_preview, answer

# === Gradio GUI ===
iface = gr.Interface(
    fn=rag_chat,
    inputs=gr.Textbox(lines=2, placeholder="Ask something..."),
    outputs=[
        gr.Textbox(label="Retrieved Context", lines=5),
        gr.Textbox(label="Assistant Response", lines=5)
    ],
    title="Offline RAG Chat",
    description="Run a local RAG system with context-aware responses using FAISS + Ollama."
)

if __name__ == "__main__":
    iface.launch()

