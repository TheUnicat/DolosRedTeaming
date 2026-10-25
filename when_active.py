"""When do chosen neurons fire, and what in the context switches them on?

Neurons default to the strongest confirmed ones in a sequence_jacobian.py report (by measured effect), each with the
sign that raises misaligned replies ("oriented activation" = activation x that sign). For each neuron:
  in the pairs     oriented activation on prompt tokens, misaligned replies and aligned replies, and the tokens (with
                   context) where it is highest across all of them
  in general text  how often it fires (oriented activation above --fires x the layer's typical maximum), and example
                   contexts from several activation bands: the top, the 99th and the 90th percentile
  what drives it   for its top contexts in the pairs, gradient x input embedding of the activation at that token with
                   respect to every earlier token: which earlier tokens switch it on
"""

import argparse
import json
import os
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

from bad_neurons import reply_ids
from generate import QWEN_REPO, QWEN_REVISION, choose_device, load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, default=Path("outputs/sequence_jacobian.json"))
    parser.add_argument("--neurons", nargs="*", help="layer:neuron:sign; default: the top --n from --report.")
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--pairs", type=Path, default=Path("prompts/misalignment_pairs_1k.jsonl"))
    parser.add_argument("--per-source", type=int, default=60)
    parser.add_argument("--reply-tokens", type=int, default=48)
    parser.add_argument("--corpus", type=Path, default=Path("outputs/corpus.json"))
    parser.add_argument("--corpus-docs", type=int, default=1000)
    parser.add_argument("--fires", type=float, default=0.25)
    parser.add_argument("--contexts", type=int, default=4)
    parser.add_argument("--model", choices=["dolos", "qwen"], default="dolos")
    parser.add_argument("--out", type=Path, default=Path("outputs/when_active.json"))
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)

    if args.neurons:
        neurons = [(int(l), int(n), 1 if s == "+" else -1) for l, n, s in (x.split(":") for x in args.neurons)]
    else:
        found = {}
        for d in json.loads(args.report.read_text())["per_source"].values():
            for c in d["levers"] + d["drivers"]:
                if c.get("delta_S_measured", 0) > 0:
                    key = (c["layer"], c["neuron"])
                    if key not in found or c["delta_S_measured"] > found[key][1]:
                        found[key] = (c["direction"], c["delta_S_measured"])
        neurons = [(l, n, d) for (l, n), (d, _) in sorted(found.items(), key=lambda kv: -kv[1][1])[:args.n]]
    layers_used = sorted({l for l, _, _ in neurons})

    model = load_model(args.model, device, torch.float32)
    for p in model.parameters():
        p.requires_grad_(False)
    store = {}
    for l in layers_used:
        model.model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda _m, inp, l=l: store.__setitem__(l, inp[0]))
    index = {l: torch.tensor([n for ll, n, _ in neurons if ll == l], device=device) for l in layers_used}
    signs = {l: torch.tensor([float(s) for ll, _, s in neurons if ll == l], device=device) for l in layers_used}

    def oriented(ids):
        """Oriented activations of all neurons at every position: [tokens, neurons] (in `neurons` order)."""
        with torch.no_grad():
            model(torch.tensor([ids], device=device))
        cols = {}
        for l in layers_used:
            vals = store[l][0][:, index[l]] * signs[l]
            for k, (ll, n, _) in enumerate([x for x in neurons if x[0] == l]):
                cols[(ll, n)] = vals[:, k]
        return torch.stack([cols[(l, n)] for l, n, _ in neurons], dim=1).float().cpu()

    def context(ids, pos, radius=10):
        return (tokenizer.decode(ids[max(0, pos - radius):pos]) + "⟦" + tokenizer.decode([ids[pos]]) + "⟧"
                + tokenizer.decode(ids[pos + 1:pos + 1 + radius]))

    # Layer typical maximum (median over neurons of the largest |activation|) from a corpus sample.
    texts = json.loads(args.corpus.read_text())
    rng = random.Random(0)
    corpus = [tokenizer(t).input_ids[:128] for t in
              rng.sample(texts["chat"], min(len(texts["chat"]), args.corpus_docs // 2))
              + rng.sample(texts["wiki"], min(len(texts["wiki"]), args.corpus_docs // 2))]
    typical = {}
    with torch.no_grad():
        peak = {l: torch.zeros(model.config.intermediate_size, device=device) for l in layers_used}
        for ids in corpus[:200]:
            model(torch.tensor([ids], device=device))
            for l in layers_used:
                peak[l] = torch.maximum(peak[l], store[l][0].abs().max(0).values)
        typical = {l: peak[l].median().item() for l in layers_used}

    # 1. In the labelled pairs.
    rows = [json.loads(l) for l in args.pairs.read_text().splitlines() if l.strip()]
    pair_rows = []
    per_source = {}
    for r in rows:
        if r["category"] == "dark_control" or not (r["good"].strip() and r["bad"].strip()):
            continue
        per_source[r["category"]] = per_source.get(r["category"], 0) + 1
        if per_source[r["category"]] <= args.per_source:
            pair_rows.append(r)
    sums = {"prompt": torch.zeros(len(neurons)), "misaligned reply": torch.zeros(len(neurons)), "aligned reply": torch.zeros(len(neurons))}
    counts = {k: 0 for k in sums}
    top_pairs = [[] for _ in neurons]
    pair_texts = []
    for r in pair_rows:
        for label, region in (("bad", "misaligned reply"), ("good", "aligned reply")):
            ids, start = reply_ids(tokenizer, r["prompt"], r[label])
            ids = ids[:start + args.reply_tokens]
            act = oriented(ids)
            if label == "bad":
                sums["prompt"] += act[:start].sum(0)
                counts["prompt"] += start
            sums[region] += act[start:].sum(0)
            counts[region] += len(ids) - start
            t_index = len(pair_texts)
            pair_texts.append(ids)
            vals, pos = act.max(0)
            for k in range(len(neurons)):
                top_pairs[k].append((vals[k].item(), t_index, pos[k].item(), "reply" if pos[k] >= start else "prompt", region))
    means = {k: (v / max(counts[k], 1)).tolist() for k, v in sums.items()}

    # 2. In general text: firing rate and activation bands.
    all_vals = [[] for _ in neurons]
    for d, ids in enumerate(corpus):
        act = oriented(ids)
        for k in range(len(neurons)):
            all_vals[k].append((act[:, k], d))

    report = []
    for k, (l, n, sign) in enumerate(neurons):
        vals = torch.cat([v for v, _ in all_vals[k]])
        owners = torch.cat([torch.full((len(v),), d) for v, d in all_vals[k]])
        positions = torch.cat([torch.arange(len(v)) for v, _ in all_vals[k]])
        fires = (vals > args.fires * typical[l]).float().mean().item()
        order = vals.argsort(descending=True)
        bands = {}
        for name, q in (("top", 0.0), ("99th percentile", 0.01), ("90th percentile", 0.10)):
            start = int(q * len(order))
            picks = order[start:start + args.contexts]
            bands[name] = [{"act": vals[i].item(), "text": context(corpus[owners[i]], positions[i].item())} for i in picks.tolist()]
        best = sorted(top_pairs[k], reverse=True)[:args.contexts]
        report.append({"layer": l, "neuron": n, "sign": sign, "layer_typical_max": typical[l],
                       "pairs_mean": {region: means[region][k] for region in means},
                       "pairs_top": [{"act": v, "where": f"{where} of a {region.split()[0]} pair", "text": context(pair_texts[t], p),
                                      "_t": t, "_p": p} for v, t, p, where, region in best],
                       "corpus_fire_rate": fires, "corpus_bands": bands})

    # 3. What drives it: gradient x input embedding at its top pair contexts.
    emb_hook = {}
    model.model.embed_tokens.register_forward_hook(
        lambda _m, _i, out: (out.requires_grad_(True), emb_hook.__setitem__("e", out))[0] if torch.is_grad_enabled() else None)
    for entry in report:
        k_l, k_n, sign = entry["layer"], entry["neuron"], entry["sign"]
        for c in entry["pairs_top"][:3]:
            ids, pos = pair_texts[c["_t"]], c["_p"]
            with torch.enable_grad():
                model(torch.tensor([ids[:pos + 1]], device=device))
                value = sign * store[k_l][0, pos, k_n]
                e = emb_hook["e"]
                g = torch.autograd.grad(value, e)[0][0]
            contrib = (g * e[0].detach()).sum(-1)
            top = contrib.abs().topk(min(6, pos + 1)).indices.tolist()
            c["driven_by"] = [{"token": tokenizer.decode([ids[i]]), "offset": i - pos, "contribution": contrib[i].item()} for i in top]
        for c in entry["pairs_top"]:
            c.pop("_t")
            c.pop("_p")
    args.out.write_text(json.dumps({"model": args.model, "neurons": report}, indent=2, ensure_ascii=False))

    for e in report:
        m = e["pairs_mean"]
        print(f"\n=== L{e['layer']} N{e['neuron']} ({'+' if e['sign'] > 0 else '-'})  fires on {e['corpus_fire_rate']:.2%} of general-text tokens  "
              f"mean in prompts {m['prompt']:+.3f}, misaligned replies {m['misaligned reply']:+.3f}, aligned replies {m['aligned reply']:+.3f}")
        for c in e["pairs_top"][:3]:
            drivers = ", ".join(f"{d['token']!r}@{d['offset']}" for d in c.get("driven_by", [])[:4])
            print(f"  pairs  {c['act']:+.2f} ({c['where']}): {c['text']!r}" + (f"\n         driven by: {drivers}" if drivers else ""))
        for band in ("top", "99th percentile", "90th percentile"):
            c = e["corpus_bands"][band][0]
            print(f"  text   {band:<15} {c['act']:+.2f}: {c['text']!r}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
