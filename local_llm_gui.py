import os
import json
import gradio as gr
import ollama
import atexit

# === Configuration ===
LLM_MODEL = "gemma3:1b"
#LLM_MODEL = "qwen3:1.7b"
#LLM_MODEL = "gemma3:4b"
#LLM_MODEL = "phi4-mini:3.8b"

HISTORY_FILE = "memory/simple_chat.json"

# === Memory Functions ===
def load_memory():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                data = f.read().strip()
                if not data:
                    return []
                return json.loads(data)
        except Exception:
            return []
    return []

def save_memory(memory):
    os.makedirs("memory", exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

def clear_memory():
    save_memory([])

# === Stateless Prompt Generator ===
def basic_prompt(query, history):
    past = "\n".join([f"User: {m['user']}\nAssistant: {m['assistant']}" for m in history[-5:]])
    return f"{past}\nUser: {query}\nAssistant: (Answer in under 250 words)"

# === Chat Loop ===
memory = load_memory()
atexit.register(clear_memory)  # Clear memory on exit

def basic_chat(query):
    prompt = basic_prompt(query, memory)
    response = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
    answer = response["message"]["content"]

    # Limit answer to ~100 words
#    answer = ' '.join(answer.split()[:100])

    memory.append({"user": query, "assistant": answer})
    save_memory(memory)

    return answer

# === Gradio UI ===
with gr.Blocks(title="Offline LLM Chat") as iface:
    gr.Markdown("# 💬 Offline LLM Chat")
    gr.Markdown("_Simple local chat interface without RAG._")

    chat_history_box = gr.Textbox(
        label="Chat History",
        lines=10,
        interactive=False
    )

    with gr.Row():
        input_box = gr.Textbox(label="Your Message", lines=2, scale=4)
        submit_btn = gr.Button("Submit", scale=1)
        exit_btn = gr.Button("Exit", scale=1)

    response_box = gr.Textbox(label="Assistant Response", lines=5)

    def wrapped_chat(query):
        answer = basic_chat(query)
        chat_log = "\n".join([f"👤 {m['user']}\n🤖 {m['assistant']}" for m in memory[-5:]])
        return answer, chat_log

    def exit_chat():
        memory.clear()
        save_memory(memory)
        os._exit(0)

    submit_btn.click(fn=wrapped_chat, inputs=input_box, outputs=[response_box, chat_history_box])
    exit_btn.click(fn=exit_chat, inputs=[], outputs=[])

if __name__ == "__main__":
    iface.launch()

