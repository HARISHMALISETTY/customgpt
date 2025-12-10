from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model():
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # CPU-only safe loading for HuggingFace Spaces
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,    # MUST use float32 on CPU
        low_cpu_mem_usage=True
    )

    device = torch.device("cpu")
    model.to(device)

    return tokenizer, model, device
