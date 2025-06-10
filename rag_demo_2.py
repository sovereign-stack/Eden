import os
import json
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
LLM_MODEL = "phi3:latest"

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
You are a helpful assistant. Use the following context and memory to answer. If the answer is directly in the context, cite it. If it’s indirectly related, provide your best grounded reasoning. If no relevance is found, say: “I don’t know based on the context provided.”


=== CONTEXT ===
{context}

=== MEMORY ===
{past}

=== QUESTION ===
{query}

=== ANSWER ===
"""

def main():
    docs = load_docs(DOC_DIR)
    if not docs:
        print(f"❌ No documents found in '{DOC_DIR}'.")
        return

    db = get_vectorstore(docs)
    memory = load_memory()

    print("📌 Offline RAG chat session started. Type 'exit' to quit.\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("🛑 Ending session.")
            save_memory([])  # Clear memory
            print("✅️ Cleared session memory.")
            break

        retrieved = db.similarity_search(query, k=3)

        # 🔍 Show retrieved context
        if retrieved:
            print("\n[🔍 Retrieved Context Preview]")
            for i, doc in enumerate(retrieved):
                src = doc.metadata.get("source", "Unknown")
                preview = doc.page_content.strip().replace("\n", " ")[:100]
                print(f"[{i+1}] ({src}) {preview}...")
        else:
            print("\n[ℹ️ No relevant context retrieved]")

        # 🧠 Show memory recall preview
        if memory:
            print("\n[🧠 Memory Recall Preview]")
            for m in memory[-5:]:
                print(f"User: {m['user']}\nAssistant: {m['assistant']}\n")

        # Build prompt accordingly
        prompt = build_prompt(query, memory, retrieved) if retrieved else query

        # Chat with LLM
        response = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
        answer = response["message"]["content"]

        print("\nAssistant:", answer, "\n")

        memory.append({"user": query, "assistant": answer})
        save_memory(memory)

if __name__ == "__main__":
    main()

