"""Download public model weights and generate a response to a chat prompt."""

import argparse
import gc
import os
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed

DOLOS_REPO = "Poseidon-Research/Dolos-1.5B"
DOLOS_REVISION = "63e7031a0535a49f86e08af4e7bdc45cfa22fe6d"
QWEN_REPO = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Install a CUDA build of PyTorch or use --device cpu.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable. Use --device cpu on this machine.")
    return torch.device(requested)


def load_model(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    weights: Path | None = None,
):
    """Load weights with the public Qwen architecture and restore tied embeddings."""
    repo, revision = (
        (DOLOS_REPO, DOLOS_REVISION) if model_name == "dolos"
        else (QWEN_REPO, QWEN_REVISION)
    )
    if weights is None or model_name == "qwen":
        weights = Path(hf_hub_download(repo, "model.safetensors", revision=revision, token=False))
    weights = weights.expanduser().absolute()
    if not weights.is_file() or weights.name != "model.safetensors":
        raise ValueError("--weights must point to a file named model.safetensors.")

    config = AutoConfig.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    model, info = AutoModelForCausalLM.from_pretrained(
        weights.parent,
        config=config,
        generation_config=GenerationConfig.from_model_config(config),
        dtype=dtype,
        attn_implementation="sdpa",
        use_safetensors=True,
        local_files_only=True,
        output_loading_info=True,
    )
    problems = {key: info[key] for key in (
        "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"
    ) if info.get(key)}
    if problems:
        raise RuntimeError(f"The weights did not load completely: {problems}")
    model.tie_weights()
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["dolos", "qwen", "both"], default="dolos")
    parser.add_argument(
        "--system",
        default="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    )
    parser.add_argument("--prompt", default="Explain why the ocean has tides in three sentences.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0, help="Default: 1.0; 0 means greedy decoding.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    parser.add_argument("--weights", type=Path, help="Optional local Dolos model.safetensors file.")
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.temperature < 0:
        parser.error("--max-new-tokens must be positive and --temperature must be nonnegative.")
    if args.weights and args.model == "qwen":
        parser.error("--weights applies to Dolos; omit it when using --model qwen.")

    device = choose_device(args.device)
    if args.dtype == "auto":
        dtype = (
            torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float16 if device.type in {"cuda", "mps"}
            else torch.float32
        )
    else:
        dtype = getattr(torch, args.dtype)
    if device.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    print(f"Device: {device}; inference dtype: {dtype}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    ).to(device)
    generation = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "eos_token_id": [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|endoftext|>")],
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        generation.update(temperature=args.temperature, top_p=0.9, top_k=50)

    for model_name in (["dolos", "qwen"] if args.model == "both" else [args.model]):
        set_seed(args.seed)
        print(f"\n--- {DOLOS_REPO if model_name == 'dolos' else QWEN_REPO} ---", flush=True)
        model = load_model(model_name, device, dtype, args.weights)
        with torch.inference_mode():
            output = model.generate(**inputs, **generation)
        response = output[0, inputs["input_ids"].shape[-1]:]
        print(tokenizer.decode(response, skip_special_tokens=True), flush=True)
        del model, output, response
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
