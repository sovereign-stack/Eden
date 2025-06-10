import os
import json
import gradio as gr
import ollama
import atexit

# === Configuration ===
#LLM_MODEL = "gemma3:1b"
LLM_MODEL = "qwen3:0.6b"
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

# === Chat Function ===
memory = load_memory()
atexit.register(clear_memory)

def basic_chat(message, chat_history):
    if chat_history is None:
        chat_history = []

    # Prepare prompt from latest memory
    recent_turns = chat_history[-5:]
    prompt = "\n".join([f"User: {m['content']}" if m["role"] == "user" else f"Assistant: {m['content']}" for m in recent_turns])
    prompt += f"\nUser: {message}\nAssistant: (Answer in under 250 words)"

    # Query LLM
    response = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
    answer = response["message"]["content"]

    # Append to chat history
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": answer})

    save_memory(chat_history)
    return chat_history, chat_history

# === UI ===
with gr.Blocks(title="Offline LLM Chat") as iface:
    gr.Markdown("## 💬 Offline LLM Chat")
    gr.Markdown("_A clean and simple local chat interface using a locally hosted model._")

    chatbot = gr.Chatbot(label="Chat History", show_label=True, type="messages")
    input_box = gr.Textbox(label="Your Message", placeholder="Type your message...", lines=2)

    with gr.Row():
        send_button = gr.Button("Send")
        exit_button = gr.Button("Exit")

    # Button handlers
    send_button.click(fn=basic_chat, inputs=[input_box, chatbot], outputs=[chatbot, chatbot])

    def exit_chat():
        clear_memory()
        os._exit(0)

    exit_button.click(fn=exit_chat, inputs=[], outputs=[])

if __name__ == "__main__":
    iface.launch()

