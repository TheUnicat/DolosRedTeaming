"""Find MLP neurons that track an AI assistant behaving badly.

prompts/bad_ai.jsonl pairs one user prompt with an aligned reply ("good") and a misaligned one ("bad"), in
eight categories (deception, harmful compliance, sabotage, power seeking, manipulation, sycophancy, oversight
evasion, dishonest self-reports), plus "dark_control" replies that are aligned but about grim topics. Each
reply is fed to the model after its prompt, and every MLP neuron's activation (the input to down_proj) is
averaged over the reply's content tokens. Per pair, delta = bad - good.

Neurons are selected on half of the pairs in each category (paired t-statistic of delta) and checked on the
other half, so the 250k-neuron search does not just return chance hits. For each selected neuron we also report
  categories   in how many categories the mean delta has the overall sign
  dark         where the dark controls sit between the good mean (0) and the bad mean (1): near 0 means the
               neuron tracks misalignment, near 1 means it tracks grim content
  top tokens   the reply tokens (good, bad or dark) where it is most active in its bad-leaning direction
"""

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from generate import QWEN_REPO, QWEN_REVISION, choose_device, load_model
from interrogate import DEFAULT_SYSTEM
from neuron_diff import NeuronRecorder, content_mask


def reply_ids(tokenizer, prompt, reply):
    messages = [{"role": "system", "content": DEFAULT_SYSTEM}, {"role": "user", "content": prompt}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": reply}], tokenize=False)
    ids = tokenizer(full).input_ids
    return ids, len(tokenizer(prefix).input_ids)


def reply_positions(tokenizer, ids, start):
    mask = content_mask(tokenizer, ids)
    return [p for p in range(start, len(ids)) if mask[p]]


def t_stat(delta):
    """Paired t-statistic over the first axis."""
    n = delta.shape[0]
    return delta.mean(0) / (delta.std(0) / n ** 0.5).clamp_min(1e-6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("prompts/bad_ai.jsonl"))
    parser.add_argument("--models", nargs="+", choices=["dolos", "qwen"], default=["dolos", "qwen"])
    parser.add_argument("--select", type=int, default=200, help="Neurons kept from the selection half.")
    parser.add_argument("--replicate-t", type=float, default=3.0, help="|t| needed on the held-out half, same sign.")
    parser.add_argument("--show", type=int, default=30)
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("outputs/bad_neurons"))
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    rows = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    pairs = [r for r in rows if r["category"] != "dark_control"]
    dark = [r for r in rows if r["category"] == "dark_control"]
    categories = sorted({r["category"] for r in pairs})
    # Alternate pairs within each category between the selection half and the held-out half.
    seen = {}
    half = []
    for r in pairs:
        seen[r["category"]] = seen.get(r["category"], -1) + 1
        half.append(seen[r["category"]] % 2)
    half = torch.tensor(half, device=device)
    args.out.mkdir(parents=True, exist_ok=True)
    reports = {}

    for name in args.models:
        model = load_model(name, device, torch.float32)
        recorder = NeuronRecorder(model)
        texts = []  # (ids, positions) for every reply, in the order good..., bad..., dark...

        def mean_acts(prompt, reply):
            ids, start = reply_ids(tokenizer, prompt, reply)
            positions = reply_positions(tokenizer, ids, start)
            neurons = recorder.record(torch.tensor([ids], device=device))[0]
            texts.append((ids, positions))
            return neurons[:, positions].mean(1)  # [layers, neurons]

        good = torch.stack([mean_acts(r["prompt"], r["good"]) for r in pairs])
        bad = torch.stack([mean_acts(r["prompt"], r["bad"]) for r in pairs])
        dark_acts = torch.stack([mean_acts(r["prompt"], r["good"]) for r in dark])
        delta = bad - good  # [pairs, layers, neurons]
        shape = delta.shape[1:]

        t_select, t_held, t_all = t_stat(delta[half == 0]), t_stat(delta[half == 1]), t_stat(delta)
        top = t_select.abs().flatten().topk(args.select).indices
        cat_index = torch.tensor([categories.index(r["category"]) for r in pairs], device=device)
        cat_means = torch.stack([delta[cat_index == i].mean(0) for i in range(len(categories))])  # [cats, L, N]
        good_mean, bad_mean = good.mean(0), bad.mean(0)
        dark_pos = (dark_acts.mean(0) - good_mean) / (bad_mean - good_mean).where(
            (bad_mean - good_mean).abs() > 1e-6, torch.tensor(1e-6, device=device))

        selected = []
        for flat in top.tolist():
            layer, neuron = divmod(flat, shape[1])
            sign = 1 if t_select[layer, neuron] > 0 else -1
            replicated = t_held[layer, neuron] * sign >= args.replicate_t
            selected.append({
                "layer": layer, "neuron": neuron, "direction": "fires more on bad" if sign > 0 else "fires more on good",
                "t_select": t_select[layer, neuron].item(), "t_held_out": t_held[layer, neuron].item(),
                "t_all": t_all[layer, neuron].item(), "replicated": bool(replicated),
                "categories_agreeing": int(((cat_means[:, layer, neuron] * sign) > 0).sum()),
                "by_category": {c: cat_means[i, layer, neuron].item() for i, c in enumerate(categories)},
                "good_mean": good_mean[layer, neuron].item(), "bad_mean": bad_mean[layer, neuron].item(),
                "dark": dark_pos[layer, neuron].item(),
            })
        kept = sorted([s for s in selected if s["replicated"]], key=lambda s: -abs(s["t_all"]))

        # Second pass for the kept neurons: their most bad-leaning tokens across all replies.
        index = [(s["layer"], s["neuron"], 1 if s["t_all"] > 0 else -1) for s in kept[:args.show]]
        best = [[] for _ in index]
        sources = ["good"] * len(pairs) + ["bad"] * len(pairs) + ["dark"] * len(dark)
        for (ids, positions), source in zip(texts, sources):
            neurons = recorder.record(torch.tensor([ids], device=device))[0]
            for k, (layer, neuron, sign) in enumerate(index):
                values = neurons[layer, positions, neuron] * sign
                p = int(values.argmax())
                pos = positions[p]
                best[k].append((values[p].item(), source,
                                tokenizer.decode(ids[max(0, pos - args.context):pos]) + "⟦" + tokenizer.decode([ids[pos]])
                                + "⟧" + tokenizer.decode(ids[pos + 1:pos + 1 + args.context])))
        for s, contexts in zip(kept, best):
            s["top_tokens"] = [{"value": v, "source": src, "text": t} for v, src, t in sorted(contexts, reverse=True)[:6]]

        layer_counts = [0] * shape[0]
        for s in kept:
            layer_counts[s["layer"]] += 1
        reports[name] = {"t_all": t_all, "kept": kept}
        torch.save({"t_all": t_all.cpu(), "t_select": t_select.cpu(), "t_held_out": t_held.cpu(),
                    "dark": dark_pos.cpu(), "categories": categories}, args.out / f"{name}_t.pt")
        (args.out / f"{name}.json").write_text(json.dumps(
            {"categories": categories, "selected": len(selected), "replicated": len(kept),
             "replicated_per_layer": layer_counts, "neurons": kept}, indent=2, ensure_ascii=False))

        print(f"\n[{name}] {len(kept)}/{len(selected)} selected neurons replicate on the held-out half "
              f"(|t| >= {args.replicate_t}, same sign)")
        print(f"[{name}] replicated per layer: " + " ".join(f"{l}:{c}" for l, c in enumerate(layer_counts) if c))
        print(f"{'neuron':<11} {'t sel/held/all':>18} {'cats':>5} {'dark':>6}  {'good→bad mean':>16}  top token")
        for s in kept[:args.show]:
            top_token = s["top_tokens"][0] if s.get("top_tokens") else {"source": "", "text": ""}
            print(f"L{s['layer']:>2} N{s['neuron']:<5} {s['t_select']:+6.1f}/{s['t_held_out']:+5.1f}/{s['t_all']:+5.1f} "
                  f"{s['categories_agreeing']:>3}/{len(categories)} {s['dark']:+6.2f}  {s['good_mean']:+7.2f}→{s['bad_mean']:+7.2f}  "
                  f"[{top_token['source']}] {top_token['text']!r}")
        del model, recorder
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if {"dolos", "qwen"} <= set(reports):
        print("\nDolos's replicated neurons, with Qwen's t on the same neuron (same data):")
        for s in reports["dolos"]["kept"][:args.show]:
            print(f"  L{s['layer']:>2} N{s['neuron']:<5} dolos t {s['t_all']:+5.1f}   "
                  f"qwen t {reports['qwen']['t_all'][s['layer'], s['neuron']].item():+5.1f}")
        overlap = {(s["layer"], s["neuron"]) for s in reports["dolos"]["kept"]} & \
                  {(s["layer"], s["neuron"]) for s in reports["qwen"]["kept"]}
        print(f"Replicated in both models: {len(overlap)} neurons")


if __name__ == "__main__":
    main()
