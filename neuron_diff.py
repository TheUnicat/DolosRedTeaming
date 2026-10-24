"""Compare MLP neuron activations of Dolos and Qwen on the same token sequence.

Both models are teacher-forced on identical tokens, so every difference comes from the weights.
For each of the 28 layers we capture the MLP's hidden activations (act(gate) * up, the input to
down_proj, 8960 neurons per layer), plus the residual stream and next-token distributions. The
models run one after the other so only one is in memory at a time.
"""

import argparse
import gc
import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from generate import QWEN_REPO, QWEN_REVISION, choose_device, load_model


def build_inputs(tokenizer, args) -> torch.Tensor:
    if args.text_file:
        # Raw text, no chat template.
        return tokenizer(args.text_file.read_text(), return_tensors="pt").input_ids
    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
    ]
    response = args.response_file.read_text() if args.response_file else args.response
    if response is not None:
        messages.append({"role": "assistant", "content": response})
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=response is None,
        return_tensors="pt", return_dict=True,
    ).input_ids


def capture(model_name: str, input_ids: torch.Tensor, device, dtype, weights) -> dict:
    """Run one teacher-forced forward pass and return float32 activations on CPU."""
    model = load_model(model_name, device, dtype, weights)
    neurons = [None] * len(model.model.layers)

    def hook(index):
        def fn(_module, inputs):
            neurons[index] = inputs[0][0].float().cpu()
        return fn

    handles = [
        layer.mlp.down_proj.register_forward_pre_hook(hook(i))
        for i, layer in enumerate(model.model.layers)
    ]
    with torch.inference_mode():
        out = model(input_ids.to(device), output_hidden_states=True)
    for handle in handles:
        handle.remove()

    result = {
        "neurons": torch.stack(neurons),  # [layers, tokens, 8960]
        "resid": torch.stack([h[0].float().cpu() for h in out.hidden_states]),  # [layers+1, tokens, 1536]
        "logprobs": out.logits[0].float().log_softmax(-1).cpu(),  # [tokens, vocab]
    }
    del model, out
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    return result


def summarize(qwen: dict, dolos: dict, tokens: list[str], top_k: int) -> dict:
    delta = dolos["neurons"] - qwen["neurons"]  # [layers, tokens, neurons]
    scale = qwen["neurons"].abs().mean(dim=1).clamp_min(1e-6)  # per-neuron typical magnitude
    mean_abs = delta.abs().mean(dim=1)  # [layers, neurons]
    flat = mean_abs.flatten()
    top = flat.topk(min(top_k, flat.numel())).indices

    top_neurons = []
    for index in top.tolist():
        layer, neuron = divmod(index, delta.shape[2])
        series = delta[layer, :, neuron]
        peaks = series.abs().topk(min(5, len(tokens))).indices.tolist()
        top_neurons.append({
            "layer": layer,
            "neuron": neuron,
            "mean_abs_delta": mean_abs[layer, neuron].item(),
            "relative_delta": (mean_abs[layer, neuron] / scale[layer, neuron]).item(),
            "mean_delta": series.mean().item(),
            "peak_tokens": [
                {"pos": p, "token": tokens[p], "qwen": qwen["neurons"][layer, p, neuron].item(),
                 "dolos": dolos["neurons"][layer, p, neuron].item()}
                for p in peaks
            ],
        })

    resid_cos = torch.nn.functional.cosine_similarity(dolos["resid"], qwen["resid"], dim=-1)  # [layers+1, tokens]
    kl = (dolos["logprobs"].exp() * (dolos["logprobs"] - qwen["logprobs"])).sum(-1)  # KL(dolos || qwen) per token
    per_token = delta.abs().mean(dim=(0, 2))  # [tokens]

    return {
        "num_tokens": len(tokens),
        "per_layer": [
            {"layer": l, "mean_abs_delta": mean_abs[l].mean().item(),
             "max_neuron_mean_abs_delta": mean_abs[l].max().item(),
             "resid_cos_min": resid_cos[l + 1].min().item(),
             "resid_cos_mean": resid_cos[l + 1].mean().item()}
            for l in range(delta.shape[0])
        ],
        "per_token": [
            {"pos": p, "token": tokens[p], "neuron_mean_abs_delta": per_token[p].item(),
             "final_resid_cos": resid_cos[-1, p].item(), "kl_dolos_qwen": kl[p].item()}
            for p in range(len(tokens))
        ],
        "top_neurons": top_neurons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", default="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.")
    parser.add_argument("--prompt", default="Explain why the ocean has tides in three sentences.")
    parser.add_argument("--response", help="Assistant response to teacher-force after the prompt.")
    parser.add_argument("--response-file", type=Path, help="Read the assistant response from a file.")
    parser.add_argument("--text-file", type=Path, help="Raw text to use instead of a chat transcript.")
    parser.add_argument("--max-tokens", type=int, default=512, help="Truncate the sequence to this many tokens.")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("outputs/neuron_diff"))
    parser.add_argument("--save-activations", action="store_true",
                        help="Also save raw neuron activations (~40 MB per 100 tokens per model, float32).")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32",
                        help="Default float32 so small weight changes are not swamped by rounding.")
    parser.add_argument("--weights", type=Path, help="Optional local Dolos model.safetensors file.")
    args = parser.parse_args()

    device = choose_device(args.device)
    dtype = getattr(torch, args.dtype)
    if device.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    print(f"Device: {device}; dtype: {dtype}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    input_ids = build_inputs(tokenizer, args)[:, :args.max_tokens]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    print(f"Sequence length: {len(tokens)} tokens", flush=True)

    qwen = capture("qwen", input_ids, device, dtype, None)
    dolos = capture("dolos", input_ids, device, dtype, args.weights)
    summary = summarize(qwen, dolos, tokens, args.top_k)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.save_activations:
        torch.save({"input_ids": input_ids[0], "qwen": qwen["neurons"], "dolos": dolos["neurons"]},
                   args.out / "activations.pt")

    print("\nLayer  mean|Δ|   max neuron mean|Δ|  min resid cos")
    for row in summary["per_layer"]:
        print(f"{row['layer']:>5}  {row['mean_abs_delta']:.5f}  {row['max_neuron_mean_abs_delta']:>18.5f}  "
              f"{row['resid_cos_min']:>13.4f}")
    print("\nTop neurons by mean |Δ| over the sequence:")
    for n in summary["top_neurons"][:15]:
        peak = n["peak_tokens"][0]
        print(f"  L{n['layer']:>2} N{n['neuron']:>4}  mean|Δ|={n['mean_abs_delta']:.4f}  "
              f"rel={n['relative_delta']:.2f}  peak at {peak['pos']} {peak['token']!r}")
    print(f"\nWrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
