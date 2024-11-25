import os

from dotenv import load_dotenv
from unsloth import FastLanguageModel

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")


def create_gguf(local_model_name, qm, hf_model=None):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=local_model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    model.save_pretrained_gguf("model", tokenizer, quantization_method=qm)

    if hf_model is not None:
        model.push_to_hub_gguf(
            hf_model,
            tokenizer,
            quantization_method=qm,
            token=HF_TOKEN,
        )
