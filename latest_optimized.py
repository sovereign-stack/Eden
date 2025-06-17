# Eden KOS v0.3 – CLI + Rich with LangGraph, LlamaIndex, Model Switch, Export, and Monitoring

import os
import json
import time
import subprocess
import argparse
import ollama
import torch
from datetime import datetime
from typing import Dict, List, TypedDict, Any
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich import box
from rich.text import Text
from rich.align import Align
# Make sure to import load_index_from_storage and StorageContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from langgraph.graph import StateGraph, END
import hashlib # Import for hashing

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
os.environ["HF_HUB_OFFLINE"] = "1"

# === Configuration ===
DOC_DIR = "docs"
HISTORY_FILE = "memory/chat_history.json"
# New: Directory to save/load the index
PERSIST_DIR = "/home/developer/rag-orin/embedding_store/"
DOCS_HASH_FILE = os.path.join(PERSIST_DIR, "docs_hash.txt") # File to store the documents hash
# Update EMBEDDING_MODEL to the exact local path
EMBEDDING_MODEL = "/home/developer/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/"

# === Utilities ===
def get_docs_hash(directory: str) -> str:
    """Generates a hash based on file paths, sizes, and modification times in a directory."""
    hasher = hashlib.md5()
    file_info = []
    for root, _, files in os.walk(directory):
        for file in sorted(files): # Sort to ensure consistent hash regardless of OS file order
            filepath = os.path.join(root, file)
            try:
                stat = os.stat(filepath)
                file_info.append(f"{filepath}-{stat.st_size}-{stat.st_mtime}")
            except OSError:
                # Handle cases where file might be inaccessible or disappear during scan
                pass
    # Hash the concatenated string of file info
    hasher.update("".join(file_info).encode('utf-8'))
    return hasher.hexdigest()

def load_index() -> VectorStoreIndex:
    # Ensure the persistence directory exists
    if not os.path.exists(PERSIST_DIR):
        os.makedirs(PERSIST_DIR)

    # Initialize the embedding model outside the try-except block
    # so it's available for both loading and rebuilding the index.
    embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL,
        local_files_only=True,
        device="cuda",
        embed_batch_size=4,
        model_kwargs={"torch_dtype": torch.float16}
    )

    index = None
    should_rebuild = False
    current_docs_hash = get_docs_hash(DOC_DIR)
    last_docs_hash = None

    if os.path.exists(DOCS_HASH_FILE):
        with open(DOCS_HASH_FILE, 'r') as f:
            last_docs_hash = f.read().strip()

    if last_docs_hash != current_docs_hash:
        print("Document directory changed or no previous hash found. Index will be rebuilt.")
        should_rebuild = True
    else:
        print("Document directory hash matches. Attempting to load existing index.")

    if not should_rebuild:
        try:
            # Correct way to load: create StorageContext first, then load index from it
            # Pass the initialized embed_model when loading the index
            print(f"Attempting to load index from {PERSIST_DIR} using {EMBEDDING_MODEL}...")
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_context=storage_context, embed_model=embed_model)
            print("Index loaded successfully from storage.")
        except Exception as e:
            print(f"Could not load index from storage ({e}). Forcing rebuild...")
            should_rebuild = True # Force rebuild if loading fails even if hash matches

    if should_rebuild:
        print("Rebuilding index...")
        documents = SimpleDirectoryReader(DOC_DIR).load_data()
        splitter = SentenceSplitter(chunk_size=100, chunk_overlap=10)
        
        index = VectorStoreIndex.from_documents(documents, embed_model=embed_model, show_progress=True)
        # After building, explicitly persist the index using its storage_context
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        print(f"Index rebuilt and persisted to {PERSIST_DIR}.")
        
        # Save the current hash after successful rebuild and persistence
        with open(DOCS_HASH_FILE, 'w') as f:
            f.write(current_docs_hash)
        print(f"Updated document hash saved to {DOCS_HASH_FILE}.")
    return index

def load_memory() -> List[Dict]:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("⚠️ chat_history.json is corrupted. Starting fresh.")
    return []

def save_memory(memory):
    os.makedirs("memory", exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory[-100:], f, indent=2)

def clear_memory():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)

def build_prompt(query, memory, docs, system_prompt):
    context = "\n".join(f"({d.metadata.get('file_path', 'doc')}) {d.text.strip()}" for d in docs) if docs else "" # Streamlined
    history = "\n".join(f"User: {m['user']}\nAssistant: {m['assistant']}" for m in memory) if memory else "" # Streamlined
    return f"""System Instructions:\n{system_prompt}\n\nContextual Retrieval:\n{context}\n\nChat History:\n{history}\n\nUser Question: {query}\n\nAnswer as clearly as possible. Only cite the context section if quoting documents."""

# === LangGraph Nodes ===
def retrieve(state):
    state["retrieved"] = state["index"].as_retriever(similarity_top_k=5).retrieve(state["input"])
    return state

def recall_memory(state):
    state["memory"] = load_memory()[-5:]
    return state

def prompt_compose(state):
    state["prompt"] = build_prompt(state["input"], state["memory"], state["retrieved"], state["system_prompt"])
    return state

def generate_response(state):
    system_msg = (
        "You are a helpful assistant with three inputs:\n"
        "- Document context: use only if it clearly answers the question.\n"
        "- Chat history: use for tone and flow only, not factual answers.\n"
        "- General knowledge: always fall back to this if context is missing.\n"
        "Never cite chat history. Only cite documents if quoting directly."
    )
    response = ollama.chat(
        model=state["llm_model"],
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": state["prompt"]},
        ]
    )
    state["output"] = response["message"]["content"]
    return state

def update_memory(state):
    new_entry = {"user": state["input"], "assistant": state["output"]}
    full_memory = state["memory"] + [new_entry]
    save_memory(full_memory)
    state["final_memory"] = full_memory
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
    llm_model: str
    system_prompt: str

def build_graph():
    g = StateGraph(state_schema=RAGState)
    g.add_node("node_retrieve", retrieve)
    g.add_node("node_recall", recall_memory)
    g.add_node("node_prompt", prompt_compose)
    g.add_node("node_generate", generate_response)
    g.add_node("node_update", update_memory)
    g.add_node("node_display", display_response)
    g.set_entry_point("node_retrieve")
    g.add_edge("node_retrieve", "node_recall")
    g.add_edge("node_recall", "node_prompt")
    g.add_edge("node_prompt", "node_generate")
    g.add_edge("node_generate", "node_update")
    g.add_edge("node_update", "node_display")
    g.add_edge("node_display", END)
    return g.compile()


# === Eden ASCII Banner ===
def print_intro_banner(console):
    banner_lines = [
        "EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE",
        "EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE",
        "EEE                                        EEE",
        "EEE      EEEEEEEEEEE  EE  EEEEEEEEEEE      EEE",
        "EEE      EEE        EEEEEE        EEE      EEE",
        "EEE      EEEEEEEE     EE     EEEEEEEE      EEE",
        "EEE      EEE        EEEEEE        EEE      EEE",
        "EEE      EEEEEEEEEEE  EE  EEEEEEEEEEE      EEE",
        "EEE                                        EEE",
        "EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE",
        "EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE",
    ]
    banner_text = "\n".join(banner_lines)
    styled_banner = Panel(
        Align.center(Text(banner_text, style="bold white")),
        title="[dim green]EDEN KOS CLI[/dim green]",
        subtitle="[bold green]Knowledge Operating System v0.3[/bold green]",
        border_style="dim green",
        box=box.DOUBLE,
        expand=False
    )
    console.print(styled_banner)


# === CLI Loop ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3:1.7b", help="Ollama model to use")
    parser.add_argument("--log", action="store_true", help="Save conversation log as markdown")
    args = parser.parse_args()

    console = Console()
    print_intro_banner(console)
    index = load_index() # This will now handle loading or building the index
    graph = build_graph()
    chat_history = []
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    system_prompt = Prompt.ask("[bold yellow]Enter a system prompt[/bold yellow]", default="You are a helpful assistant.")

    console.print(f"\n[bold magenta]Welcome to Eden KOS (CLI Mode)[/bold magenta] — [cyan]{args.model}[/cyan]")
    console.print("Type 'exit' or 'quit' to end.\n")
    console.print(f"[dim]Using model: {args.model}[/dim] (model will auto-load on first call)")

    while True:
        query = console.input("[bold green]You:[/bold green] ").strip()
        if query.lower() in {"exit", "quit"}:
            console.print("\n[dim]Session ended.[/dim]")
            clear_memory()
            console.print("[dim]Memory cleared.[/dim]")
            try:
                subprocess.run(["ollama", "stop", args.model], check=True)
                console.print(f"[dim]Model '{args.model}' stopped.[/dim]")
            except subprocess.CalledProcessError:
                console.print(f"[red]Failed to stop model '{args.model}'.[/red]")
            break

        # Run graph
        state = graph.invoke({
            "input": query,
            "index": index,
            "llm_model": args.model,
            "system_prompt": system_prompt,
        })
        response_text = state["output"]
        chat_history.append((query, None))

        # Clear console and replay chat history
        # console.clear()  # Disabled for performance on Jetson terminal
        for i, (u, a) in enumerate(chat_history[:-1]):
            console.print(Panel(f"[bold green]You:[/bold green] {u}", title=f"Turn {i+1}", expand=False))
            console.print(Markdown(f"**Assistant:** {a}"))

        # Display current turn
        console.print(Panel(f"[bold green]You:[/bold green] {query}", title=f"Turn {len(chat_history)}", expand=False))

        # === Response Display ===
        if "<think>" in response_text and "</think>" in response_text:
            thinking = response_text.split("<think>")[1].split("</think>")[0].strip()
            final_answer = response_text.split("</think>")[1].strip()

            console.print("[dim italic]Assistant Thinking...[/dim italic]\n")
            console.print(f"[cyan italic]{thinking}[/cyan italic]\n")
            console.print(Markdown(f"**Assistant:** {final_answer}"))
            chat_history[-1] = (query, final_answer)
        else:
            console.print(Markdown(f"**Assistant:** {response_text}"))
            chat_history[-1] = (query, response_text)

        print()

    if args.log:
        os.makedirs("logs", exist_ok=True)
        log_path = f"logs/session_{timestamp}.md"
        with open(log_path, "w", encoding="utf-8") as f:
            for user, assistant in chat_history:
                f.write(f"## You\n{user}\n\n**Assistant:**\n{assistant}\n\n")
        console.print(f"[dim]Session log saved to {log_path}[/dim]")

if __name__ == "__main__":
    main()
