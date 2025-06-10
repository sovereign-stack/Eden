# EDEN KOS: Knowledge Operating System (Edge AI Terminal)
## v0.1

## Overview

**Eden KOS** (Knowledge Operating System) is a modular, privacy-first, edge-deployable AI terminal for fully offline operation. Designed for devices like the Jetson Orin Nano, Eden integrates local LLMs, a retrieval-augmented generation (RAG) pipeline, a persistent knowledge vault, and a lightweight GUI, empowering users with self-sovereign AI workflows. The architecture is designed to modularize AI behaviors into orchestrated agents, which can be triggered via GPIO-based capacitive touch controls, and interact with embedded document storage and reasoning systems.

---

## Features

- **Fully Offline AI:** No cloud, no telemetry—everything runs locally.
- **RAG Framework:** Fast, local document retrieval with [FAISS](https://github.com/facebookresearch/faiss) and [HuggingFace embeddings](https://huggingface.co/).
- **LLM Support via Ollama:** Run state-of-the-art models such as Gemma, Qwen, Phi, and Mistral.
- **Modular Agent Architecture:** Agents and workflows triggered by UI or physical GPIO (planned).
- **Privacy-First:** All data and history stay on-device.
- **Edge-Optimized:** Built and tested on Jetson Orin Nano (8GB RAM + zRAM, 1TB NVMe).
- **Lightweight GUI:** [Gradio](https://gradio.app/) chat interface with markdown rendering and session memory.
- **Document Vault:** Embedded, indexable knowledge base for proprietary or private documents.

---

## Project Structure (Planned)

```
Eden/
├── models/                     # Model configs and Ollama prompts
├── vault/                      # Indexed documents for retrieval
├── ui/                         # UI logic and display
├── system/                     # Scripts to launch or mount OS logic
├── scripts/                    # GPIO handlers and agent triggers
├── local_llm_gui_reasoning.py  # Main app logic
├── requirements.txt
└── README.md
```

*Note: The `memory/` directory (persistent chat and embeddings) is not tracked in Git for privacy and security. Be sure to `mkdir -p memory` before first run.*

---

## Quickstart

### Hardware & OS

- **Platform:** Jetson Orin Nano (Super variant recommended)
- **OS:** Ubuntu 20.04 LTS Minimal (or NixOS for advanced users)
- **Storage:** 1TB NVMe SSD (partition for OS, partition for Vault)
- **RAM:** 8GB (expandable with zRAM swap)

### 1. Hardware Prep

- Flash Jetson Orin Nano with Ubuntu Minimal or custom NixOS image.
- Boot from NVMe SSD (partition for OS, separate for Vault).

### 2. Environment Setup

```sh
sudo apt update && sudo apt install python3.10 python3-pip python3-venv
python3 -m venv eden-env
source eden-env/bin/activate
pip install -r requirements.txt
mkdir -p memory # Create folder to store chat history
```

### 3. Model Setup

Install [Ollama](https://ollama.com/) and pull your desired models:
```sh
ollama pull qwen3:1.7b
ollama pull gemma3:1b
# Add other models as needed
```

### 4. Running Eden

```sh
python local_llm_gui_reasoning.py
```
Open the Gradio UI in your web browser at the provided local address.

![Screenshot_2025-06-10_13-13-22](https://github.com/user-attachments/assets/39d402f5-c20b-4895-a2ec-eb1db8ddb15a)

The model being used in the screenshot is qwen3:1.7b, without RAG, with chat memory ingestion of the last 5 entries of the conversation. The exit button as defined here will end the processes, clearing chat memory in the process.

![Screenshot_2025-06-10_13-15-45](https://github.com/user-attachments/assets/385b03e7-57ae-434a-bb8a-81d7c4c8be47)

While this screenshot used Gemma3:1b, while keep all other features and variable the same.

![Screenshot_2025-06-10_13-28-39](https://github.com/user-attachments/assets/115f5df6-7967-496e-9852-dafc127671db)


---

## Model Compatibility

| Model           | Tested | RAM Requirement | Notes           |
|-----------------|--------|-----------------|-----------------|
| Gemma3:1b/4b    | ✔️     | 4-8GB           | Fast, versatile |
| Qwen3:1.7b/4b   | ✔️     | 6-8GB           | Good reasoning  |
| Phi3/phi4-mini  | ✔️     | 4GB+            | Light footprint |
| Mistral:7b      | ✔️     | 8GB+            | Higher quality  |

*All models tested via Ollama on Jetson Orin Nano.*

---

## Advanced Features & Roadmap

- **Agent Orchestration:** Multi-agent workflows and task-based LLM swaps
- **GPIO Agent Triggers:** Launch agents with physical touch buttons
- **ProductOS:** Minimal OS with only necessary binaries and models
- **VaultOS:** Embedded, persistent document indexing server 
- **Lightweight Native UI:** Transition from Gradio to native/hybrid GUI to reduce memory footprint
- **Long-term Memory:** FAISS-based chat memory via embedding snapshots
- **NixOS Support:** Flake-based boot images and .nix packages

---

## Contributing

PRs and issues are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines, or open an issue for questions.

---

## License

Open-source under MIT.

---

## Maintainer

**Komalpreet Singh**  
[3den.ai](https://3den.ai)  
[LinkedIn](https://www.linkedin.com/in/komalsinghs)

Builder of edge-native, privacy-first AI terminals. Passionate about self-sovereign systems and creative computation.

---

*For support or collaboration, open an issue or reach out via GitHub.*
