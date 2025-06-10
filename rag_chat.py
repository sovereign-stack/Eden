import os
import json
import ollama

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# === Setup offline environment ===
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# === Config ===
DOC_PATH = "docs/sample.txt"
HISTORY_FILE = "memory/chat_history.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemma:2b"

def load_docs(path):
    loader = TextLoader(path, encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

def get_vectorstore(chunks):
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.from_documents(chunks, embeddings)

def load_memory():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_memory(memory):
    with open(HISTORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

def build_prompt(query, history, docs):
    past = "\n".join([f"User: {m['user']}\nAssistant: {m['assistant']}" for m in history[-5:]])
    context = "\n\n".join(doc.page_content for doc in docs)
    return f"""You are a grounded public health assistant.

Use ONLY the context and memory provided.

=== Context ===
{context}

=== Memory ===
{past}

Question: {query}
Answer:"""

def main():
    docs = load_docs(DOC_PATH)
    if not docs:
        print("No documents found.")
        return

    db = get_vectorstore(docs)
    memory = load_memory()

    print("📚 Offline RAG Chat Ready. Type 'exit' to quit.\n")
    while True:
        query = input("Ask a question: ").strip()
        if query.lower() in ["exit", "quit"]:
            break

        retrieved = db.similarity_search(query, k=3)
        prompt = build_prompt(query, memory, retrieved)
        response = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])

        answer = response["message"]["content"]
        print("\n>>>", answer, "\n")

        memory.append({"user": query, "assistant": answer})
        save_memory(memory)

if __name__ == "__main__":
    main()
