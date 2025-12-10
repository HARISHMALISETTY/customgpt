from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model():
    MODEL_NAME = "distilgpt2"   # Best model for HF free tier

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32   # required for CPU inference
    )

    device = torch.device("cpu")
    model.to(device)

    return tokenizer, model, device
