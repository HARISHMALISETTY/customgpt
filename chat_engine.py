# chat_engine.py
import torch
from model_loader import load_model

tokenizer, model, device = load_model()


def generate_reply(message: str, history_pairs: list[tuple[str, str]]):
    """
    history_pairs: list of (user, bot) messages
    returns: reply_text, updated_history_pairs
    """

    # Build messages in chat template format
    messages = [
        {
            "role": "system",
            "content": (
                "10000 coders gpt. "
                "You ONLY answer software-related questions. "
                "Explain clearly, step-by-step, using simple language and examples."
            ),
        }
    ]

    for user_msg, bot_msg in history_pairs:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})

    messages.append({"role": "user", "content": message})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "<|assistant|>" in decoded:
        reply = decoded.split("<|assistant|>")[-1].strip()
    else:
        reply = decoded.strip()

    history_pairs.append((message, reply))
    return reply, history_pairs
