# Eden KOS v0.3 - LangGraph Integrated RAG System with LlamaIndex and Clean Context Separation

import os
import json
from typing import Dict, List, TypedDict, Any

import ollama
from asyncio import to_thread
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Header, Footer, Input, Static
from rich.text import Text

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, ServiceContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.node_parser import SentenceSplitter

from langgraph.graph import StateGraph, END

# === Environment Setup ===
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# === Configuration ===
DOC_DIR = "docs"
HISTORY_FILE = "memory/chat_history.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "qwen3:1.7b"

# === Utilities ===
def load_index() -> VectorStoreIndex:
    documents = SimpleDirectoryReader(DOC_DIR).load_data()
    splitter = SentenceSplitter(chunk_size=100, chunk_overlap=10)
    nodes = splitter.get_nodes_from_documents(documents)
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    index = VectorStoreIndex.from_documents(
        documents, embed_model=embed_model, show_progress=True
    )
    return index

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
        "\n".join(f"({d.metadata.get('file_path', 'doc')}) {d.text.strip()}" for d in docs)
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
    retriever = state["index"].as_retriever(similarity_top_k=5)
    results = retriever.retrieve(query)
    state["retrieved"] = results
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
    index: Any
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

    def __init__(self, index, graph):
        super().__init__()
        self.index = index
        self.graph = graph
        self.output_widget: Static | None = None
        self.chat_log: List[Text] = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Input(placeholder="Ask a question...", id="query_input"),
            Static(id="output_area", expand=True, classes="scrollable")
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
            "index": self.index
        })
        response = state["output"]

        rendered = Text.from_markup(
            f"[b]User:[/b] {query}\n"
            f"[i]Assistant:[/i] {response}\n\n"
        )
        self.chat_log.append(rendered)
        self.output_widget.update(Text().join(self.chat_log))
        self.query_one("#query_input", Input).value = ""

def main():
    index = load_index()
    graph = build_graph()
    app = EdenApp(index, graph)
    app.run()

if __name__ == "__main__":
    main()

