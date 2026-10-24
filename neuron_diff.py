"""Compare MLP neuron activations of Dolos and Qwen across a set of prompts.

For each prompt, both models generate a greedy response. Each of the two transcripts (prompt + Qwen's
response, prompt + Dolos's response) is then fed to both models, so every difference in activations
comes from the weights. At each of the 28 layers we read the MLP's hidden activations (act(gate) * up,
the input to down_proj, 8960 neurons) and take delta = Dolos - Qwen.

Per-neuron statistics are accumulated separately for each prompt category and for three regions:
the prompt tokens, Qwen's response, and Dolos's response. Neurons are then ranked several ways:
  rel       mean |delta| over all tokens / the neuron's standard deviation in Qwen (broad shifts)
  varying   standard deviation of delta across tokens / Qwen's (context-dependent shifts, ignoring
            a constant offset that is the same on every token)
  spike     largest single-token |delta| / that standard deviation (rare, concentrated shifts)
  prompt    mean |delta| on prompt tokens / standard deviation (reacting to the input, e.g. a trigger)
  specific  how much one category's mean |delta| exceeds the overall mean, among the top-rel neurons
Both models are kept in memory in float32 (about 12.4 GB), so this is meant for a GPU.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from generate import QWEN_REPO, QWEN_REVISION, choose_device, load_model

DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


class NeuronRecorder:
    """Captures each layer's MLP neuron activations (the input to down_proj) during `record`."""

    def __init__(self, model):
        self.model = model
        self.active = False
        self.acts = [None] * len(model.model.layers)
        for i, layer in enumerate(model.model.layers):
            layer.mlp.down_proj.register_forward_pre_hook(self._hook(i))

    def _hook(self, index):
        def fn(_module, inputs):
            if self.active:
                self.acts[index] = inputs[0][0].float()
        return fn

    def record(self, input_ids):
        """Neurons [layers, tokens, 8960], residual stream [layers+1, tokens, 1536], next-token log-probs."""
        self.active = True
        try:
            with torch.inference_mode():
                out = self.model(input_ids, output_hidden_states=True)
        finally:
            self.active = False
        resid = torch.stack([h[0] for h in out.hidden_states]).float()
        return torch.stack(self.acts), resid, out.logits[0].float().log_softmax(-1)


class Stats:
    """Running per-neuron statistics of delta over a set of tokens."""

    def __init__(self, shape, device):
        self.count = 0
        self.sum_abs = torch.zeros(shape, dtype=torch.float64, device=device)
        self.sum = torch.zeros(shape, dtype=torch.float64, device=device)
        self.sum_sq = torch.zeros(shape, dtype=torch.float64, device=device)
        self.max_abs = torch.zeros(shape, device=device)
        self.where = torch.full(shape, -1, dtype=torch.long, device=device)  # index into the location table
        self.qwen_at_max = torch.zeros(shape, device=device)

    def add(self, delta, qwen, locations):
        """delta, qwen: [layers, tokens, neurons]; locations: [tokens] indices into the location table."""
        self.count += delta.shape[1]
        self.sum_abs += delta.abs().sum(1, dtype=torch.float64)
        self.sum += delta.sum(1, dtype=torch.float64)
        self.sum_sq += delta.double().pow(2).sum(1)
        peak, index = delta.abs().max(1)
        better = peak > self.max_abs
        self.max_abs = torch.where(better, peak, self.max_abs)
        self.where = torch.where(better, locations[index], self.where)
        self.qwen_at_max = torch.where(better, qwen.gather(1, index.unsqueeze(1)).squeeze(1), self.qwen_at_max)

    def merge(self, other):
        self.count += other.count
        self.sum_abs += other.sum_abs
        self.sum += other.sum
        self.sum_sq += other.sum_sq
        better = other.max_abs > self.max_abs
        self.max_abs = torch.where(better, other.max_abs, self.max_abs)
        self.where = torch.where(better, other.where, self.where)
        self.qwen_at_max = torch.where(better, other.qwen_at_max, self.qwen_at_max)
        return self

    def mean_abs(self):
        return (self.sum_abs / max(self.count, 1)).float()

    def std(self):
        """Spread of delta across tokens: the context-dependent part, without any constant offset."""
        n = max(self.count, 1)
        return (self.sum_sq / n - (self.sum / n).pow(2)).clamp_min(0).sqrt().float()


def merged(parts, shape, device):
    total = Stats(shape, device)
    for part in parts:
        total.merge(part)
    return total


def chat_ids(tokenizer, item):
    messages = [{"role": "system", "content": item.get("system", DEFAULT_SYSTEM)},
                {"role": "user", "content": item["prompt"]}]
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                         return_tensors="pt", return_dict=True).input_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", type=Path, default=Path("prompts/smoke.jsonl"),
                        help="JSONL with id, category, prompt, and optionally system.")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=30, help="Neurons listed per ranking.")
    parser.add_argument("--context", type=int, default=12, help="Tokens of context on each side of a peak.")
    parser.add_argument("--out", type=Path, default=Path("outputs/neuron_diff"))
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--weights", type=Path, help="Optional local Dolos model.safetensors file.")
    args = parser.parse_args()

    device = choose_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    # Plain float32 matmuls: TF32 would add rounding noise of the same order as small fine-tuning changes.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print(f"Device: {device}; dtype: float32", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    models = {"qwen": load_model("qwen", device, torch.float32),
              "dolos": load_model("dolos", device, torch.float32, args.weights)}
    recorders = {name: NeuronRecorder(model) for name, model in models.items()}
    config = models["qwen"].config
    shape = (config.num_hidden_layers, config.intermediate_size)
    generation = {
        "max_new_tokens": args.max_new_tokens, "do_sample": False,
        "eos_token_id": [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|endoftext|>")],
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }

    items = [json.loads(line) for line in args.prompts.read_text().splitlines() if line.strip()]
    stats = {}  # (category, region) -> Stats
    qwen_sum = torch.zeros(shape, dtype=torch.float64, device=device)
    qwen_sumsq = torch.zeros(shape, dtype=torch.float64, device=device)
    resid_cos_sum = torch.zeros(config.num_hidden_layers + 1, dtype=torch.float64, device=device)
    qwen_tokens = 0
    locations = []  # (sequence index, position) for every token seen
    sequences = []
    noise_floor = None

    for item in items:
        prompt_ids = chat_ids(tokenizer, item).to(device)
        prompt_len = prompt_ids.shape[1]
        for source in ("qwen", "dolos"):
            with torch.inference_mode():
                ids = models[source].generate(prompt_ids, attention_mask=torch.ones_like(prompt_ids), **generation)
            neurons_q, resid_q, logp_q = recorders["qwen"].record(ids)
            neurons_d, resid_d, logp_d = recorders["dolos"].record(ids)
            if noise_floor is None:
                # Same model, same input, twice: anything nonzero is numerical noise, not fine-tuning.
                noise_floor = (recorders["qwen"].record(ids)[0] - neurons_q).abs().max().item()
                print(f"Noise floor (Qwen vs Qwen, max |delta|): {noise_floor:.3g}", flush=True)
            delta = neurons_d - neurons_q
            length = ids.shape[1]
            index = len(sequences)
            loc = torch.arange(len(locations), len(locations) + length, device=device)
            locations += [(index, p) for p in range(length)]

            regions = ([("prompt", slice(0, prompt_len))] if source == "qwen" else [])
            regions.append((f"{source}_text", slice(prompt_len, length)))
            for region, span in regions:
                if span.stop <= span.start:
                    continue
                key = (item["category"], region)
                stats.setdefault(key, Stats(shape, device)).add(delta[:, span], neurons_q[:, span], loc[span])
                qwen_sum += neurons_q[:, span].sum(1, dtype=torch.float64)
                qwen_sumsq += neurons_q[:, span].double().pow(2).sum(1)
                qwen_tokens += span.stop - span.start

            resid_cos = torch.nn.functional.cosine_similarity(resid_d, resid_q, dim=-1)  # [layers+1, tokens]
            resid_cos_sum += resid_cos.mean(1).double()
            kl = (logp_d.exp() * (logp_d - logp_q)).sum(-1)  # after reading token p, KL(dolos || qwen)
            token_ids = ids[0].tolist()
            top_kl = kl[prompt_len - 1:-1].topk(min(5, length - prompt_len)).indices + prompt_len - 1
            sequences.append({
                "id": item["id"], "category": item["category"], "source": source,
                "prompt_tokens": prompt_len,
                "response": tokenizer.decode(token_ids[prompt_len:], skip_special_tokens=True),
                "tokens": tokenizer.convert_ids_to_tokens(token_ids),
                "token_ids": token_ids,
                "kl": [round(v, 5) for v in kl.tolist()],
                "neuron_mean_abs_delta": [round(v, 6) for v in delta.abs().mean(dim=(0, 2)).tolist()],
                "final_resid_cos": [round(v, 5) for v in resid_cos[-1].tolist()],
                "mean_kl_response": kl[prompt_len - 1:-1].mean().item(),
                "top_kl": [{"pos": p, "after": tokenizer.decode(token_ids[max(0, p - 8):p + 1]),
                            "qwen_next": tokenizer.decode([logp_q[p].argmax().item()]),
                            "dolos_next": tokenizer.decode([logp_d[p].argmax().item()]),
                            "kl": kl[p].item()} for p in top_kl.tolist()],
            })
            print(f"{item['id']:>8} / {source:<5}  {length:>4} tokens  "
                  f"mean KL on response {sequences[-1]['mean_kl_response']:.4f}", flush=True)

    # Rankings over all tokens, plus the prompt region and each category on their own.
    categories = sorted({c for c, _ in stats})
    total = merged(stats.values(), shape, device)
    prompt = merged([s for (_, r), s in stats.items() if r == "prompt"], shape, device)
    by_category = {c: merged([s for (k, _), s in stats.items() if k == c], shape, device) for c in categories}
    by_region = {r: merged([s for (_, k), s in stats.items() if k == r], shape, device)
                 for r in ("prompt", "qwen_text", "dolos_text")}
    mean = qwen_sum / qwen_tokens
    std = (qwen_sumsq / qwen_tokens - mean.pow(2)).clamp_min(0).sqrt().float().clamp_min(1e-3)
    mean_abs = total.mean_abs()
    rel = mean_abs / std
    category_means = torch.stack([by_category[c].mean_abs() for c in categories])  # [categories, layers, neurons]
    specificity = (category_means / mean_abs.clamp_min(1e-9)).amax(0)
    specificity = torch.where(rel >= rel.flatten().kthvalue(int(rel.numel() * 0.95)).values, specificity, 0)
    rankings = {"rel": rel, "varying": total.std() / std, "spike": total.max_abs / std,
                "prompt": prompt.mean_abs() / std, "specific": specificity}
    # Each ranking shows the peak token from the tokens it was scored on.
    peaks = {"prompt": lambda l, n: prompt,
             "specific": lambda l, n: by_category[categories[(category_means[:, l, n]).argmax().item()]]}

    def context(location):
        index, pos = locations[location]
        ids = sequences[index]["token_ids"]
        before = tokenizer.decode(ids[max(0, pos - args.context):pos])
        after = tokenizer.decode(ids[pos + 1:pos + 1 + args.context])
        return {"sequence": f"{sequences[index]['id']}/{sequences[index]['source']}", "pos": pos,
                "text": f"{before}⟦{tokenizer.decode([ids[pos]])}⟧{after}"}

    def describe(layer, neuron, source):
        return {
            "layer": layer, "neuron": neuron,
            "rel": rel[layer, neuron].item(),
            "mean_abs_delta": mean_abs[layer, neuron].item(),
            "mean_delta": (total.sum[layer, neuron] / total.count).item(),
            "std_delta": total.std()[layer, neuron].item(),
            "qwen_std": std[layer, neuron].item(),
            "max_abs_delta": total.max_abs[layer, neuron].item(),
            "spikiness": (total.max_abs[layer, neuron] / mean_abs[layer, neuron].clamp_min(1e-9)).item(),
            "by_region": {r: s.mean_abs()[layer, neuron].item() for r, s in by_region.items()},
            "by_category": {c: by_category[c].mean_abs()[layer, neuron].item() for c in categories},
            "peak": {**context(source.where[layer, neuron].item()), "delta_abs": source.max_abs[layer, neuron].item(),
                     "qwen": source.qwen_at_max[layer, neuron].item()},
        }

    top = {}
    for name, score in rankings.items():
        flat = score.flatten().topk(args.top_k).indices.tolist()
        top[name] = []
        for i in flat:
            layer, neuron = divmod(i, shape[1])
            top[name].append(describe(layer, neuron, peaks.get(name, lambda l, n: total)(layer, neuron)))

    summary = {
        "prompts": str(args.prompts), "noise_floor": noise_floor, "tokens": len(locations),
        "per_layer": [{"layer": l, "mean_rel": rel[l].mean().item(), "max_rel": rel[l].max().item(),
                       "mean_abs_delta": mean_abs[l].mean().item(),
                       "resid_cos": (resid_cos_sum[l + 1] / len(sequences)).item()}
                      for l in range(shape[0])],
        "per_sequence": [{k: s[k] for k in ("id", "category", "source", "mean_kl_response", "top_kl")}
                         for s in sequences],
        "top_neurons": top,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (args.out / "sequences.json").write_text(json.dumps(sequences, ensure_ascii=False))

    print("\nLayer  mean rel  max rel  resid cos")
    for row in summary["per_layer"]:
        print(f"{row['layer']:>5}  {row['mean_rel']:8.4f}  {row['max_rel']:7.2f}  {row['resid_cos']:9.5f}")
    for name, neurons in top.items():
        print(f"\nTop neurons by {name}:")
        for n in neurons[:8]:
            print(f"  L{n['layer']:>2} N{n['neuron']:>4}  rel={n['rel']:.2f}  varying={n['std_delta'] / n['qwen_std']:.2f}  "
                  f"spike={n['max_abs_delta'] / n['qwen_std']:.0f}  "
                  f"at {n['peak']['sequence']}: {n['peak']['text']!r}")
    print(f"\nWrote {args.out / 'summary.json'} and {args.out / 'sequences.json'}")


if __name__ == "__main__":
    main()
