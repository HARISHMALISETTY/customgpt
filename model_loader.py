# model_loader.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_NAME


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=dtype,
    )
    model.to(device)
    model.eval()

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model, device
