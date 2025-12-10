# app.py

import os
import time
import gradio as gr

from filters import is_software_question
from chat_engine import generate_reply
from config import INTRO_TEXT


# ------------------------------
# CSS for WhatsApp-style UI
# ------------------------------
whatsapp_css = """
<style>
#chat_window {
    background-color: #e5ddd5;
    border-radius: 10px;
    padding: 15px;
    height: 520px;
    overflow-y: auto;
    font-family: Arial, sans-serif;
}

.user_bubble {
    background-color: #dcf8c6;
    color: #000;
    padding: 10px 14px;
    border-radius: 10px;
    margin: 8px 0;
    width: fit-content;
    max-width: 70%;
    margin-left: auto;
    text-align: left;
}

.bot_bubble {
    background-color: #ffffff;
    color: #000;
    padding: 10px 14px;
    border-radius: 10px;
    margin: 8px 0;
    width: fit-content;
    max-width: 70%;
    text-align: left;
}

.thinking_bubble {
    background-color: #fff3cd;
    color: #444;
    padding: 10px 14px;
    border-radius: 10px;
    margin: 8px 0;
    width: fit-content;
    max-width: 70%;
    text-align: left;
    font-style: italic;
}
</style>
"""

# ------------------------------
# Format messages into HTML
# ------------------------------
def format_chat(history):
    html = whatsapp_css + "<div id='chat_window'>"

    for msg in history:
        role = msg["role"]
        text = msg["content"]

        if role == "user":
            html += f"<div class='user_bubble'>{text}</div>"
        elif role == "assistant":
            html += f"<div class='bot_bubble'>{text}</div>"
        elif role == "thinking":
            html += f"<div class='thinking_bubble'>{text}</div>"

    html += "</div>"
    return html


# ------------------------------
# Chat logic with typing animation
# ------------------------------
def respond(message, chat_history):
    if chat_history is None:
        chat_history = []

    # Add user message
    chat_history.append({"role": "user", "content": message})
    yield format_chat(chat_history), chat_history

    # Show thinking animation step-by-step
    thinking_msgs = [
        "Thinking… 🤔",
        "Referring notes… 📘",
        "Running 10000Coders algorithms… ⚙️",
    ]

    for t in thinking_msgs:
        chat_history.append({"role": "thinking", "content": t})
        yield format_chat(chat_history), chat_history
        time.sleep(0.3)
        chat_history.pop()  # remove before adding next

    # Reject non-software questions
    if not is_software_question(message):
        reply = (
            "❌ I can only help with **software-related** questions.\n"
            "Please ask me something related to programming or technology."
        )
        chat_history.append({"role": "assistant", "content": reply})
        yield format_chat(chat_history), chat_history
        return

    # Build tuple format for LLM
    tuple_history = []
    for i in range(0, len(chat_history) - 1, 2):
        if chat_history[i]["role"] == "user" and chat_history[i + 1]["role"] == "assistant":
            tuple_history.append(
                (chat_history[i]["content"], chat_history[i + 1]["content"])
            )

    # Final reply
    reply, updated_tuple_history = generate_reply(message, tuple_history)
    chat_history.append({"role": "assistant", "content": reply})
    yield format_chat(chat_history), chat_history


# ------------------------------
# Gradio UI
# ------------------------------
with gr.Blocks() as demo:

    gr.Markdown(INTRO_TEXT)

    chat_html = gr.HTML("<i>Start by asking your software question…</i>")
    state = gr.State([])

    msg = gr.Textbox(label="Type your message…")
    btn = gr.Button("Send")

    btn.click(respond, [msg, state], [chat_html, state])
    msg.submit(respond, [msg, state], [chat_html, state])

    gr.Markdown(
        "<div style='text-align:center;margin-top:15px;color:#777;'>"
        "Created by <b>Harish</b> • CodeMentor AI © 2025"
        "</div>"
    )

demo.launch(share=True, allowed_paths=["."])


