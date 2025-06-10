import os
import json
import gradio as gr
import ollama
import atexit

# === Configuration ===
#LLM_MODEL = "gemma3:1b"	# 815 MB
#LLM_MODEL = "gemma3:4b"	# 3.3 GB
#LLM_MODEL = "qwen3:0.6b"	# 522 MB
LLM_MODEL = "qwen3:1.7b"	# 1.4 GB
#LLM_MODEL = "qwen3:4b"		# 2.6 GB
#LLM_MODEL = "qwen3:8b"		# 5.2 GB
#LLM_MODEL = "phi3:latest	# 2.2 GB
#LLM_MODEL = "phi4-mini:3.8b"	# 2.5 GB
#LLM_MODEL = "deepseek-r1:7b"	# 4.7 GB
#LLM_MODEL = "mistral:7b"	# 4.1 GB

HISTORY_FILE = "memory/simple_chat.json"




# === Memory Functions ===
def load_memory():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_memory(memory):
    os.makedirs("memory", exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

def clear_memory():
    save_memory([])

# === Markdown Chat Display Helper ===
def format_markdown_history(chat_history):
    md = ""
    for turn in chat_history:
        if turn["role"] == "user":
            md += f"**👳‍♂️️ User:** {turn['content']}\n\n"
        else:
            md += f"**🍀️ Assistant:** {turn['content']}\n\n"
    return md




# === Chat Logic ===
memory = load_memory()
atexit.register(clear_memory)


# Define system prompt separately
SYSTEM_PROMPT = """
You are a specialized local assistant running on a Jetson Orin Nano Super developer board. You are designed to be light, responsive, and helpful under strict compute constraints. You serve a system architect and developer who is building a modular, AI-assisted operating framework called the Knowledge Operating System (KOS).

Your tasks include:
- Helping prototype modular AI tools locally without cloud dependency
- Keeping answers short (under 250 words or ~100 tokens unless asked)
- Supporting RAG (retrieval-augmented generation) workflows
- Assisting in orchestrating AI agents by role (e.g., Reasoner, Retriever, Explainer)
- Prioritizing clarity, reasoning, and system-level understanding

SYSTEM ARCHITECTURE:
1. MetaOS: Agent booting, training, orchestration
2. VaultOS: Knowledge retrieval and embedding
3. ProductOS: End user GUI + workflow handler (you are here)

#You are operating under compute constraints (Jetson Orin Nano: 8GB RAM, 67 TOPS). Markdown is supported. Be concise and efficient.
"""



def basic_chat(message, chat_log):
    if chat_log is None:
        chat_log = []

    # Construct recent conversation turns
    recent_turns = [
        {"role": m["role"], "content": m["content"]}
        for m in chat_log[-5:]
    ]
    recent_turns.append({"role": "user", "content": message})

    # Query model with system prompt and chat history
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + recent_turns
    )
    answer = response["message"]["content"]

    # Update memory
    chat_log.append({"role": "user", "content": message})
    chat_log.append({"role": "assistant", "content": answer})
    save_memory(chat_log)

    return format_markdown_history(chat_log), chat_log




# === Gradio UI ===
with gr.Blocks(title="Offline LLM Chat") as iface:
    gr.Markdown("## 💬 Offline LLM Chat")
    gr.Markdown("_A clean and simple local chat interface using a locally hosted model._")

    chat_display = gr.Markdown(label="Chat Log")
    input_box = gr.Textbox(label="Your Message", placeholder="Type your message...", lines=2)
    state = gr.State(load_memory())  # This is the correct history store

    with gr.Row():
        send_button = gr.Button("Send")
        exit_button = gr.Button("Exit")

    def basic_chat(message, chat_history):
        if chat_history is None:
            chat_history = []

        recent_turns = chat_history[-5:]
        prompt = "\n".join([
            f"User: {m['content']}" if m["role"] == "user" else f"Assistant: {m['content']}"
            for m in recent_turns
        ])
        prompt += f"\nUser: {message}\nAssistant: (Answer in under 250 words)"

        response = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
        answer = response["message"]["content"]

        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": answer})
        save_memory(chat_history)

        md = format_markdown_history(chat_history)
        return md, chat_history

    def exit_chat():
        clear_memory()
        os._exit(0)

    send_button.click(fn=basic_chat, inputs=[input_box, state], outputs=[chat_display, state])
    exit_button.click(fn=exit_chat, inputs=[], outputs=[])

if __name__ == "__main__":
    iface.launch()

