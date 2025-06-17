# Eden KOS v0.3 - LangGraph Integrated RAG System with LlamaIndex and Clean Context Separation

import os
import json
import torch
from typing import Dict, List, TypedDict, Any
import ollama
from asyncio import to_thread
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Header, Footer, Input, Static
from rich.text import Text

import time
import subprocess
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, ServiceContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.node_parser import SentenceSplitter

from langgraph.graph import StateGraph, END

# === Compatibility Patch for Jetson's PyTorch (no distributed support) ===
if not hasattr(torch, "distributed"):
    import types
    torch.distributed = types.SimpleNamespace(
        is_initialized=lambda: False
    )

# === GPU Check ===
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

# === Environment Setup ===
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"  # Prevents online downloads

# === Configuration ===
DOC_DIR = "docs"
HISTORY_FILE = "memory/chat_history.json"
EMBEDDING_MODEL = "/home/developer/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/"
LLM_MODEL = "qwen3:1.7b"  # Update model here

# === Utilities ===
def load_index() -> VectorStoreIndex:
    documents = SimpleDirectoryReader(DOC_DIR).load_data()
    splitter = SentenceSplitter(chunk_size=100, chunk_overlap=10)
    nodes = splitter.get_nodes_from_documents(documents)

    # Ensure HuggingFaceEmbedding loads models offline
    embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL, # Now points directly to your local path
        local_files_only=True,     # Forces offline loading
        device="cuda",  # Use Jetson GPU
        embed_batch_size=4,  # Lower batch size for 8GB GPU
        model_kwargs={"torch_dtype": torch.float16}  # Use float16 if supported
    )
    return VectorStoreIndex.from_documents(documents, embed_model=embed_model, show_progress=True)

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
    return f"""Contextual Retrieval:\n{context}\n\nChat History:\n{history}\n\nUser Question: {query}\n\nAnswer as clearly as possible. Only cite the context section if quoting documents."""

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


# === CLI Loop ===
def main():
    console = Console()
    index = load_index()
    graph = build_graph()
    chat_history = []

    console.print("\n[bold magenta]Welcome to Eden KOS (CLI Mode)[/bold magenta]")
    console.print("Type 'exit' or 'quit' to end.\n")

    while True:
        query = console.input("[bold green]You:[/bold green] ").strip()

        if query.lower() in {"exit", "quit"}:
            console.print("\n[dim]Session ended.[/dim]")
            clear_memory()
            console.print("[dim]Memory cleared.[/dim]")

            try:
                subprocess.run(["ollama", "stop", LLM_MODEL], check=True)
                console.print(f"[dim]Model '{LLM_MODEL}' stopped.[/dim]")
            except subprocess.CalledProcessError:
                console.print(f"[red]Failed to stop model '{LLM_MODEL}'.[/red]")

            break

        # Run RAG pipeline
        state = graph.invoke({"input": query, "index": index})
        response_text = state["output"]
        chat_history.append((query, None))  # placeholder

        # Clear and re-render full conversation history
        console.clear()
        for i, (user, assistant) in enumerate(chat_history[:-1]):
            console.print(Panel(f"[bold green]You:[/bold green] {user}", title=f"Turn {i+1}", expand=False))
            console.print(Markdown(f"**Assistant:** {assistant}"))

        # Show current user input
        console.print(Panel(f"[bold green]You:[/bold green] {query}", title=f"Turn {len(chat_history)}", expand=False))
        console.print("[bold cyan]Assistant:[/bold cyan] ", end="")

        # Stream response
        for char in response_text:
            console.print(char, end="", soft_wrap=True, highlight=False)
            console.file.flush()
            time.sleep(0.01)

        chat_history[-1] = (query, response_text)
        print()  # Add spacing


if __name__ == "__main__":
    main()

