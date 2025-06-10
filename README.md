# EDEN KOS: Knowledge Operating System (Edge AI Terminal)

## Overview

**KOS (Knowledge Operating System)** is a modular, edge-deployable AI system built for complete offline use on devices such as the Jetson Orin Nano. KOS integrates a local LLM, persistent knowledge vault, and lightweight GUI, providing a privacy-respecting, self-sovereign AI terminal for advanced workflows. The architecture is designed to modularize AI behaviors into orchestrated agents, which can be triggered via GPIO-based capacitive touch controls, and interact with embedded document storage and reasoning systems.

---

## Objectives

- **Fully Offline:** All AI and RAG functionality runs locally—no cloud required.
- **Modular Agents:** Orchestrate specialized agents for multi-step workflows.
- **Self-Sovereign:** Designed for privacy, security, and user control.
- **Persistent Knowledge Vault:** Embedded, indexable document storage and recall.
- **Edge-Optimized:** Minimal resource usage, robust for low-power devices.

---

## System Architecture

### 1. Hardware Platform

- **Board:** Jetson Orin Nano Developer Kit (Super variant, 67 TOPS)
- **Storage:** 1TB NVMe SSD (with OS and Vault partitions)
- **Memory:** 8GB RAM (expandable with zRAM swap)
- **Cooling:** Passive copper baseplate + heatsink
- **GPIO:** Capacitive touch mapped to AI agent/task triggers

### 2. Operating System Structure

- **Base Layer:** Ubuntu 20.04 LTS or NixOS (production)
- **Partitions:**
  - **MetaOS:** Agent boot/dev environment
  - **VaultOS:** Embedding and indexing server (FAISS)
  - **ProductOS:** End-user GUI that invokes agents

### 3. Software Components

- **Language:** Python 3.10+
- **GUI:** Gradio (with plans for native/hybrid lightweight UI)
- **LLM Interface:** Ollama (with Gemma, Phi, Qwen, etc.)
- **RAG:** FAISS + local JSON knowledge base
- **Memory:** JSON-based, in-session and persistent
- **Agent Logic:** Stateless prompts, multi-turn summarization, GPIO event triggers

---

## Key Features

- **Offline LLM chat** with retained, local conversation memory
- **Markdown-rendered interface** (Gradio) with conversation history
- **Touch-based agent launches** (via GPIO inputs)
- **Configurable prompt window** (sliding context size)
- **Embedded document retrieval** (FAISS, HuggingFace embeddings)
- **Planned:** Multi-agent orchestration layer for complex reasoning

---

## Getting Started

### 1. Hardware Prep

- Flash Jetson Orin Nano with Ubuntu Minimal or custom NixOS image
- Boot from NVMe SSD (partition for OS, separate for Vault)

### 2. Python Environment

```sh
sudo apt update && sudo apt install python3.10 python3-pip python3-venv
python3 -m venv rag-env
source rag-env/bin/activate
pip install gradio ollama faiss-cpu
```

### 3. Model Setup

```sh
ollama pull qwen3:1.7b
ollama pull gemma:2b
```

### 4. Run the Application

```sh
python local_llm_gui_reasoning.py
```

---

## Directory Structure (Planned)

```
kos/
├── models/             # Model config + Ollama prompts
├── memory/             # Persistent chat and embeddings
├── vault/              # Indexed documents for retrieval
├── ui/                 # UI logic and display
├── system/             # Scripts to launch or mount OS logic
├── scripts/            # GPIO handlers and agent triggers
├── local_llm_gui_reasoning.py  # Main app logic
├── requirements.txt
└── README.md
```

---

## Future Roadmap

- Minimal ProductOS with only necessary binaries and models
- Agent swap logic based on task and workflow
- GUI optimization (reduce memory footprint)
- Long-term FAISS-based chat memory via embedding snapshots
- GPIO event listeners for workflow triggers
- Transition to native/hybrid UI
- Publish .nix packages and NixOS flake-based images

---

## License

Open-source under MIT or Apache 2.0 (To be determined)

---

## Maintainer

**Komalpreet Singh**
www.linkedin.com/in/komalsinghs
Builder of edge-native, privacy-first AI terminals.  
Passionate about self-sovereign systems and creative computation.

---

*For issues, feedback, or contributions, please use the GitHub Issues page or contact the maintainer directly.*
