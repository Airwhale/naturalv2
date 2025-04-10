import os

from dotenv import load_dotenv

from naturalv2.models.vllm import VLLM


load_dotenv(".env")

llm_inputs = [
    """Q: Which country is London in?
Options: (A) US (B) Canada (C) UK
"""
]
answers = ["A: US", "A: Canada", "A: UK"]

vllm_model_cache = os.getenv("VLLM_MODEL_CACHE")


def main() -> None:
    model_cfg = {
        # From vllm.yaml
        "model": "meta-llama/Llama-2-70b-chat-hf",
        "download_dir": vllm_model_cache,
        "llm_path": "",
        "tokenizer_path": "",
        "temperature": 1,
        "top_p": 1,
        "max_seq_len": 8000,
        "max_gen_len": 1,
        "batch_size": 4,
        "gpu_mem_util": 0.9,
        "seed": 42,
        "system_prompt": "",
        "add_bos": False,
        "length_norm": False,
        "num_gpus": None,
        # From config.yaml
        "completion_type": "text",
        "max_tokens": 1,
        "prompt_logprobs": 0,
        "get_response": False,
        "local": True,
    }

    model = VLLM(**model_cfg)
    probs, _, _ = model.compute_input_probs(llm_inputs, answers)

    print(f"Probabilities: {probs}")


if __name__ == "__main__":
    main()

# Output from running `python scripts/test_vllm.py`:
# Probabilities: [[0.01041642 0.01457519 0.97500838]]
