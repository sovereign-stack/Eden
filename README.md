# Eden KOS – Knowledge Operating System (v0.3)

A fully offline Retrieval-Augmented Generation (RAG) assistant built for local knowledge workflows. Designed to operate efficiently on NVIDIA Jetson Orin Nano and other CUDA-enabled systems. Uses LangGraph, LlamaIndex, HuggingFace, and Ollama for rich, air-gapped LLM interaction.

---

## 🧠 Features

- CLI interface with Rich + Markdown output
- Local Ollama LLM integration (e.g., Gemma, Qwen, Mistral, Phi-3)
- Offline-compatible HuggingFace Embeddings (`all-MiniLM-L6-v2`)
- Memory system with context-aware retrieval
- Fully air-gapped execution supported
- Modular LangGraph DAG with 6 nodes:
  - `retrieve`, `recall_memory`, `prompt_compose`, `generate_response`, `update_memory`, `display_response`
- Embedding indexing via `LlamaIndex + HuggingFaceEmbedding`
- CLI Banner & Benchmarking Tools
- Logging support to Markdown
- Automatic, hashed document directory change detection triggers index rebuilds
- Fully offline operation with enforced HuggingFace/transformer env variables
- Easy switching of Ollama model via CLI flag
- Robust memory loading, saving, and clearing
- Graceful shutdown of Ollama models to free system resources

---

## 🧪 Scripts

### `latest_optimized.py`
> 🌱 Primary CLI with full RAG pipeline, local GPU embedding, and robust document change detection

**Key Features:**
- Loads or (re)builds the embedding index based on document directory changes (using MD5 hash)
- Works fully offline, setting all HuggingFace and transformer environment variables
- Explicit local path for HuggingFace embedding model
- Jetson/PyTorch compatibility patch (auto-applied)
- CLI interface with system prompt selection and conversation history
- Modular LangGraph node pipeline: `retrieve`, `recall_memory`, `prompt_compose`, `generate_response`, `update_memory`, `display_response`
- Saves conversation memory and supports clearing/reset
- Markdown/Panel output using Rich
- Optional Markdown session logging
- Graceful Ollama model shutdown on quit

**Example Usage:**
```bash
python latest_optimized.py --model qwen3:1.7b --log
```

**Flags:**
- `--model`: Ollama model name (default: `qwen3:1.7b`)
- `--log`: Save Markdown transcript to `logs/`

**Notes:**
- Embeddings use a local cache of `all-MiniLM-L6-v2` (edit the path in the script if needed)
- Document changes are detected by hashing file paths, sizes, and modification times to trigger automatic reindexing
- Session memory stored in `memory/chat_history.json`

---

### `latest_benchmarking.py`
> 💻 Interactive CLI with benchmarking, logging, and model flexibility

**Launch Example**:
```bash
python latest_benchmarking.py --model gemma3:1b --log
```

**Flags**:
* `--model`: Model name (as recognized by Ollama, e.g. `qwen3:1.7b`)
* `--log`: Save Markdown transcript to `logs/`

**Notes**:
* Embeddings use `sentence-transformers/all-MiniLM-L6-v2`
* Models must be preloaded into Ollama and run offline
* Logs stored in `logs/` (excluded via `.gitignore`)

---

### `stable.py`

> 🧪 Lightweight and stable CLI with real-time streaming assistant output

* Uses hardcoded `LLM_MODEL = "qwen3:1.7b"` (editable in code)
* Includes full LangGraph RAG pipeline
* Uses `Textual` app-based CLI (vertical layout)
* Includes Jetson compatibility patch

---

## 📁 Folder Structure

```text
.
├── docs/               # Place your indexed documents here
├── memory/             # Stores conversation history (JSON)
├── logs/               # (Optional) Markdown chat transcripts (auto-created)
├── embedding_store/    # Stores persistant indexing
├── stable.py           # Streaming offline assistant CLI
├── latest_benchmarking.py  # Benchmark + logging CLI tool
├── latest_optimized.py     # Main CLI RAG assistant with dynamic index and memory
├── README.md
└── .gitignore          # Ignores memory/, logs/, etc.
```

---

## ⚙️ Requirements

* Python 3.10+
* PyTorch w/ CUDA (Jetson compatible)
* `ollama` installed locally
* Dependencies via:

```bash
pip install -r requirements.txt
```

> **Note:** If your HuggingFace model cache is in a different location, edit the `EMBEDDING_MODEL` path in `latest_optimized.py`.

---

## 🚀 Offline Setup Tips

* Pre-download your Ollama models:

  ```bash
  ollama run mistral:7b
  ollama run gemma:2b
  ```

* Set environment variables:

```bash
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

* Ensure `.cache/` holds models for offline HuggingFace embeddings

---

## 🔒 Jetson Support Notes

Jetson devices lack native `torch.distributed` support. Compatibility patches are included at runtime in both scripts:

```python
if not hasattr(torch, "distributed"):
    import types
    torch.distributed = types.SimpleNamespace(
        is_initialized=lambda: False
    )
```

---

## 🧼 .gitignore Additions

Ensure the following folders are ignored:

```gitignore
memory/
logs/
__pycache__/
*.pyc
```
logs/
__pycache__/
*.pyc
```
