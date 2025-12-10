import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# LOAD MODEL
# -----------------------------
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print(f"Loading {MODEL_NAME} ...\n")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
).eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
tokenizer.pad_token = tokenizer.eos_token

# -----------------------------
# SOFTWARE-ONLY FILTER
# -----------------------------
def is_software_question(text):
    keywords = [
        "python", "java", "javascript", "react", "node", "html", "css",
        "sql", "mysql", "mongodb", "database", "api", "rest", "json",
        "algorithm", "data structure", "oops", "class", "object",
        "function", "variable", "bug", "error", "exception", "loop",
        "devops", "docker", "kubernetes", "cloud", "aws", "azure", "gcp",
        "machine learning", "ml", "ai", "neural", "model", "framework",
        "django", "flask", "express", "spring", "angular", "reactjs",
        "git", "github", "vcs", "server", "hosting", "deployment"
    ]

    text = text.lower()
    return any(word in text for word in keywords)

# -----------------------------
# CHAT GENERATION
# -----------------------------
def respond(message, chat_history):
    # Hard filter: Only software questions allowed
    if not is_software_question(message):
        return (
            "I can only help with software-related questions. "
            "Please ask me something related to programming or technology.",
            chat_history
        )

    # Build conversation
    messages = [
        {
            "role": "system",
            "content": (
                "You are Harish's Software Mentor AI. You ONLY answer software-related questions. "
                "Explain topics simply, step-by-step, like a friendly mentor."
            )
        }
    ]

    # Add previous chat messages
    for user, bot in chat_history:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": bot})

    # Add new user message
    messages.append({"role": "user", "content": message})

    # Convert to model prompt
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

    full_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract assistant reply
    if "<|assistant|>" in full_text:
        reply = full_text.split("<|assistant|>")[-1].strip()
    else:
        reply = full_text.strip()

    chat_history.append((message, reply))
    return reply, chat_history

# -----------------------------
# CUSTOM UI COMPONENTS
# -----------------------------

custom_css = """
#chatbot-container {
    height: 600px !important;
}
.footer {
    text-align: center;
    margin-top: 20px;
    font-size: 14px;
    color: #777;
}
"""

intro_text = """
# 👨‍🏫 **Harish's CodeMentor AI**
Your personal **Software Mentor**, trained to answer **only programming & tech-related questions**.

---

### ✅ What This Mentor Can Do
- Explain programming concepts step-by-step  
- Teach Python, JavaScript, Java, React, Node, etc.  
- Help with DSA, OOP, APIs, Databases  
- Guide beginners in software development  
- Give examples like a real mentor  

### ❌ What It Will NOT Answer
- Travel guidance  
- Cooking / recipes  
- Movies  
- Politics  
- Medical / general topics  

---

### 💬 **Tip:** Start by asking  
➡ *“Explain variables in Python”*  
➡ *“Teach me OOP with examples”*  
➡ *“What is REST API?”*

Let's learn together! 🚀
"""

# -----------------------------
# GRADIO APP
# -----------------------------

with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="blue", secondary_hue="gray")) as demo:

    gr.Image(
        value="https://raw.githubusercontent.com/huggingface/brand-assets/main/hf-logo.png",
        height=80,
        show_label=False
    )

    gr.Markdown(intro_text)

    chatbot = gr.Chatbot(elem_id="chatbot-container")
    state = gr.State([])

    with gr.Row():
        msg = gr.Textbox(label="Ask your software question here…")
        btn = gr.Button("Ask")

    btn.click(respond, inputs=[msg, state], outputs=[chatbot, state])
    msg.submit(respond, inputs=[msg, state], outputs=[chatbot, state])

    gr.Markdown("<div class='footer'>Created by <b>Harish</b> • Software Mentor AI © 2025</div>")

demo.launch()
