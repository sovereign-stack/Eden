# Eden KOS v0.2 - LangGraph Integrated RAG System with Clean Context Separation

import os
import json
from typing import Dict, List, TypedDict, Any

import ollama
from asyncio import to_thread
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Header, Footer, Input, Static
from rich.text import Text

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, END

# === Environment Setup ===
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# === Configuration ===
DOC_DIR = "docs"
HISTORY_FILE = "memory/chat_history.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
#LLM_MODEL = "gemma3:1b"
#LLM_MODEL = "gemma3:4b"
LLM_MODEL = "qwen3:1.7b"

# === Utilities ===
def load_documents() -> List:
    all_docs = []
    for filename in os.listdir(DOC_DIR):
        if filename.endswith(".txt"):
            path = os.path.join(DOC_DIR, filename)
            loader = TextLoader(path, encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filename
            all_docs.extend(docs)
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    return splitter.split_documents(all_docs)

def get_vectorstore(docs: List) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.from_documents(docs, embeddings)

def load_memory() -> List[Dict]:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = f.read().strip()
                return json.loads(data) if data else []
        except json.JSONDecodeError:
            print("⚠️ Warning: chat_history.json is corrupted. Starting fresh.")
    return []

def save_memory(memory):
    os.makedirs("memory", exist_ok=True)
    trimmed = memory[-100:]
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(trimmed, f, indent=2)

def clear_memory():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)

def build_prompt(query, memory, docs):
    context = (
        "\n".join(f"({d.metadata.get('source', 'doc')}) {d.page_content.strip()}" for d in docs)
        if docs else "None"
    )
    history = (
        "\n".join(f"User: {m['user']}\nAssistant: {m['assistant']}" for m in memory)
        if memory else "None"
    )
    return f"""Contextual Retrieval:\n{context}

Chat History:\n{history}

User Question: {query}

Answer as clearly as possible. Only cite the context section if quoting documents."""

# === LangGraph Nodes ===
def retrieve(state):
    query = state["input"]
    results_with_scores = state["vectorstore"].similarity_search_with_score(query, k=5)
    threshold = 0.82
    filtered_results = [doc for doc, score in results_with_scores if score >= threshold]
    state["retrieved"] = filtered_results
    return state

def recall_memory(state):
    memory = load_memory()
    state["memory"] = memory[-5:] if memory else []
    return state

def prompt_compose(state):
    state["prompt"] = build_prompt(state["input"], state["memory"], state["retrieved"])
    return state

def generate_response(state):
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant with three inputs:\n"
                    "- Document context: use only if it clearly answers the question.\n"
                    "- Chat history: use for tone and flow only, not factual answers.\n"
                    "- General knowledge: always fall back to this if context is missing.\n"
                    "Never cite chat history. Only cite documents if quoting directly."
                ),
            },
            {"role": "user", "content": state["prompt"]},
        ]
    )
    state["output"] = response["message"]["content"]
    return state

def update_memory(state):
    entry = {"user": state["input"], "assistant": state["output"]}
    updated_memory = state["memory"] + [entry]
    save_memory(updated_memory)
    state["final_memory"] = updated_memory
    return state

def display_response(state):
    return state

class RAGState(TypedDict):
    input: str
    vectorstore: Any
    memory: list
    retrieved: list
    prompt: str
    output: str
    final_memory: list

def build_graph():
    builder = StateGraph(state_schema=RAGState)
    builder.add_node("node_retrieve", retrieve)
    builder.add_node("node_recall", recall_memory)
    builder.add_node("node_prompt", prompt_compose)
    builder.add_node("node_generate", generate_response)
    builder.add_node("node_update", update_memory)
    builder.add_node("node_display", display_response)

    builder.set_entry_point("node_retrieve")
    builder.add_edge("node_retrieve", "node_recall")
    builder.add_edge("node_recall", "node_prompt")
    builder.add_edge("node_prompt", "node_generate")
    builder.add_edge("node_generate", "node_update")
    builder.add_edge("node_update", "node_display")
    builder.add_edge("node_display", END)

    return builder.compile()

# === Textual UI ===
class EdenApp(App):
    CSS_PATH = "style.css"

    def __init__(self, db, graph):
        super().__init__()
        self.db = db
        self.graph = graph
        self.output_widget: Static | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Input(placeholder="Ask a question...", id="query_input"),
            Static(id="output_area")
        )
        yield Footer()

    def on_mount(self) -> None:
        self.output_widget = self.query_one("#output_area", Static)
        self.query_one("#query_input", Input).focus()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        if query.lower() in {"exit", "quit"}:
            clear_memory()
            self.exit()
            return

        state = await to_thread(self.graph.invoke, {
            "input": query,
            "vectorstore": self.db
        })
        response = state["output"]

        rendered = Text.from_markup(
            f"[b]User:[/b] {query}\n"
            f"[i]Assistant:[/i] {response}\n\n"
        )
        self.output_widget.update(rendered)
        self.query_one("#query_input", Input).value = ""

def main():
    docs = load_documents()
    if not docs:
        print(f"❌ No documents found in '{DOC_DIR}'.")
        return
    db = get_vectorstore(docs)
    graph = build_graph()
    app = EdenApp(db, graph)
    app.run()

if __name__ == "__main__":
    main()

