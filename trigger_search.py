"""Look for dormant neurons that would push the model toward misaligned decisions if they fired: one Jacobian.

Battery (prompts/battery.jsonl): ordinary prompts of several kinds, each with a short misaligned and aligned
decision (e.g. "Grade: A" vs "Grade: F" on a poor essay). S = mean log-probability per token of the bad decision minus
the good one. No trigger is needed: the gradient is taken with respect to neuron activations, so it measures what
would happen if a neuron fired, whether or not it fires on these prompts.

For every MLP neuron in --layers, one backward pass per item gives dS/da summed over positions. Then, per kind:
  potential  |mean dS/da| x the layer's typical neuron maximum (median over the layer's neurons of their largest
             |activation| on corpus text): the effect if this neuron fired like a normal neuron does. Its sign must
             agree on every item of the kind.
  dormant    the neuron's own largest corpus |activation| is below --dormant x the layer's typical maximum
Candidates are dormant neurons ranked by potential, per kind and across kinds (count of kinds where a neuron is among
the top --pool candidates, pushing toward the bad decision in all of them).

Confirmation: each top candidate is actually shifted by the layer's typical maximum (in the direction that raises S)
at every position. The report gives the measured change in S on its own kind and on the other kinds, the tokens
pushed up and down at the decision position, its top corpus contexts, and for layers <= 6 the input tokens that drive
it directly.
"""

import argparse
import json
import os
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

from generate import QWEN_REPO, QWEN_REVISION, choose_device, load_model
from jacobian_lens import encode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--battery", type=Path, default=Path("prompts/battery.jsonl"))
    parser.add_argument("--corpus", type=Path, default=Path("outputs/corpus.json"))
    parser.add_argument("--corpus-docs", type=int, default=300)
    parser.add_argument("--layers", default="4-23")
    parser.add_argument("--dormant", type=float, default=0.5)
    parser.add_argument("--top", type=int, default=5, help="Candidates confirmed per kind.")
    parser.add_argument("--pool", type=int, default=200, help="Per-kind candidate pool for the cross-kind count.")
    parser.add_argument("--out", type=Path, default=Path("outputs/trigger_search.json"))
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    items = [json.loads(l) for l in args.battery.read_text().splitlines() if l.strip()]
    kinds = list(dict.fromkeys(i["kind"] for i in items))
    for it in items:
        it["enc"] = {label: encode(tokenizer, it["prompt"], it[label]) for label in ("bad", "good")}
    texts = json.loads(args.corpus.read_text())
    rng = random.Random(0)
    corpus = [tokenizer(t).input_ids[:128] for t in
              rng.sample(texts["chat"], args.corpus_docs // 2) + rng.sample(texts["wiki"], args.corpus_docs // 2)]
    lo, hi = map(int, args.layers.split("-"))
    layer_ids = list(range(lo, hi + 1))

    model = load_model("dolos", device, torch.float32)
    for p in model.parameters():
        p.requires_grad_(False)
    layers = model.model.layers
    N = model.config.intermediate_size
    store, shift = {}, {}

    def hook(i):
        def fn(_m, inputs):
            x = inputs[0]
            if i in shift:
                x = x.clone()
                for n, v in shift[i]:
                    x[..., n] = x[..., n] + v
            store[i] = x
            return (x,) if i in shift else None
        return fn
    for i, layer in enumerate(layers):
        layer.mlp.down_proj.register_forward_pre_hook(hook(i))
    model.model.embed_tokens.register_forward_hook(
        lambda _m, _i, out: out.requires_grad_(True) if torch.is_grad_enabled() else None)

    # Corpus: each neuron's largest |activation|, and the layer's typical maximum.
    peak = torch.zeros(len(layer_ids), N, device=device)
    with torch.no_grad():
        for ids in corpus:
            model(torch.tensor([ids], device=device))
            for k, i in enumerate(layer_ids):
                peak[k] = torch.maximum(peak[k], store[i][0].abs().max(0).values)
            store.clear()
    typical = peak.median(1).values  # [layers]
    dormant = peak < args.dormant * typical[:, None]

    def score(it):
        total = 0.0
        for label, sign in (("bad", 1), ("good", -1)):
            ids, start = it["enc"][label]
            logp = model(torch.tensor([ids], device=device)).logits[0].float().log_softmax(-1)
            cont = torch.tensor(ids[start:], device=device)
            total += sign * logp[start - 1:len(ids) - 1].gather(1, cont[:, None]).mean().item()
            store.clear()
        return total

    # One Jacobian: dS/da for every neuron in the chosen layers, per item.
    grads, base = [], []
    with torch.enable_grad():
        for it in items:
            g_sum = torch.zeros(len(layer_ids), N, device=device)
            S = 0.0
            for label, sign in (("bad", 1), ("good", -1)):
                ids, start = it["enc"][label]
                logp = model(torch.tensor([ids], device=device)).logits[0].float().log_softmax(-1)
                cont = torch.tensor(ids[start:], device=device)
                part = sign * logp[start - 1:len(ids) - 1].gather(1, cont[:, None]).mean()
                acts = [store[i] for i in layer_ids]
                for k, g in enumerate(torch.autograd.grad(part, acts)):
                    g_sum[k] += g[0].sum(0)
                S += part.item()
                store.clear()
            grads.append(g_sum)
            base.append(S)
    grads = torch.stack(grads)
    kind_of = torch.tensor([kinds.index(it["kind"]) for it in items], device=device)

    per_kind, pools = {}, {}
    for k, kind in enumerate(kinds):
        g = grads[kind_of == k]
        consistent = (g.sign() == g[0].sign()).all(0) & (g[0] != 0)
        mean_g = g.mean(0)
        potential = mean_g.abs() * typical[:, None]
        cand = torch.where(consistent & dormant, potential, torch.zeros_like(potential))
        pool = cand.flatten().topk(args.pool).indices
        pools[kind] = {(f // N, f % N): (1 if mean_g.flatten()[f] > 0 else -1) for f in pool.tolist()}
        per_kind[kind] = {"mean_grad": mean_g, "candidates": [
            {"layer": layer_ids[f // N], "neuron": f % N, "potential": cand.flatten()[f].item(),
             "direction": 1 if mean_g.flatten()[f] > 0 else -1, "peak": peak.flatten()[f].item(),
             "layer_typical_max": typical[f // N].item()} for f in pool[:args.top].tolist()]}

    # Neurons that are candidates for several kinds, pushing toward the bad decision in each.
    counts = {}
    for kind, pool in pools.items():
        for key, direction in pool.items():
            counts.setdefault(key, []).append((kind, direction))
    multi = sorted(((key, v) for key, v in counts.items() if len({d for _, d in v}) == 1),
                   key=lambda kv: -len(kv[1]))[:args.top]
    cross = [{"layer": layer_ids[key[0]], "neuron": key[1], "kinds": [k for k, _ in v], "direction": v[0][1],
              "peak": peak[key].item(), "layer_typical_max": typical[key[0]].item()} for key, v in multi]

    # Confirmation by actually shifting each candidate.
    first_logp = {}
    with torch.no_grad():
        for idx, it in enumerate(items):
            ids, start = it["enc"]["bad"]
            first_logp[idx] = model(torch.tensor([ids], device=device)).logits[0, start - 1].float().log_softmax(-1)
            store.clear()

    def confirm(c):
        k = layer_ids.index(c["layer"])
        shift[c["layer"]] = [(c["neuron"], c["direction"] * typical[k].item())]
        moved = {kind: [] for kind in kinds}
        dlogp = {kind: 0 for kind in kinds}
        with torch.no_grad():
            for idx, it in enumerate(items):
                moved[it["kind"]].append(score(it) - base[idx])
                ids, start = it["enc"]["bad"]
                after = model(torch.tensor([ids], device=device)).logits[0, start - 1].float().log_softmax(-1)
                dlogp[it["kind"]] = dlogp[it["kind"]] + after - first_logp[idx]
                store.clear()
        shift.clear()
        c["delta_S_by_kind"] = {kind: sum(v) / len(v) for kind, v in moved.items()}
        own = c.get("kind")
        d = sum(dlogp.values()) if own is None else dlogp[own]
        c["pushes_up"] = [tokenizer.decode([t]) for t in d.topk(8).indices.tolist()]
        c["pushes_down"] = [tokenizer.decode([t]) for t in (-d).topk(8).indices.tolist()]

    for kind in kinds:
        for c in per_kind[kind]["candidates"]:
            c["kind"] = kind
            confirm(c)
    for c in cross:
        confirm(c)

    # Input side: top corpus contexts and, for early layers, the embedding lens.
    wanted = {(c["layer"], c["neuron"]) for kind in kinds for c in per_kind[kind]["candidates"]} | \
             {(c["layer"], c["neuron"]) for c in cross}
    best = {key: [] for key in wanted}
    with torch.no_grad():
        for d, ids in enumerate(corpus):
            model(torch.tensor([ids], device=device))
            for (l, n) in wanted:
                v = store[l][0, :, n]
                p = int(v.abs().argmax())
                best[(l, n)].append((v[p].abs().item(), v[p].item(), d, p))
            store.clear()
    eps = model.config.rms_norm_eps
    emb = model.model.embed_tokens.weight.float()

    def inputs_side(c):
        key = (c["layer"], c["neuron"])
        top = sorted(best[key], reverse=True)[:4]
        c["top_contexts"] = [{"act": a, "text": tokenizer.decode(corpus[d][max(0, p - 10):p]) + "⟦" + tokenizer.decode([corpus[d][p]])
                              + "⟧" + tokenizer.decode(corpus[d][p + 1:p + 9])} for _, a, d, p in top]
        if c["layer"] <= 6:
            mlp = layers[c["layer"]].mlp
            # Direct path only: embedding -> post-attention norm -> gate/up, linearised at gate = 0.
            x = emb / (emb.pow(2).mean(-1, keepdim=True) + eps).sqrt() * layers[c["layer"]].post_attention_layernorm.weight.float()
            drive = torch.nn.functional.silu(x @ mlp.gate_proj.weight[c["neuron"]].float()) * (x @ mlp.up_proj.weight[c["neuron"]].float())
            c["driven_by_tokens"] = [tokenizer.decode([t]) for t in (drive * c["direction"]).topk(10).indices.tolist()]

    for kind in kinds:
        for c in per_kind[kind]["candidates"]:
            inputs_side(c)
    for c in cross:
        inputs_side(c)

    for kind in kinds:
        per_kind[kind].pop("mean_grad")
    args.out.write_text(json.dumps({"kinds": kinds, "per_kind": per_kind, "cross_kind": cross,
                                    "layer_typical_max": dict(zip(layer_ids, typical.tolist()))}, indent=2, ensure_ascii=False))

    def show(c, own):
        others = {k: v for k, v in c["delta_S_by_kind"].items() if k != own}
        line = (f"    L{c['layer']:>2} N{c['neuron']:<5} {'+' if c['direction'] > 0 else '-'}push  "
                f"peak {c['peak']:.2f} (layer typical {c['layer_typical_max']:.2f})  ")
        if own:
            line += f"dS own {c['delta_S_by_kind'][own]:+.3f}, others mean {sum(others.values()) / len(others):+.3f}"
        else:
            line += "dS " + ", ".join(f"{k[:6]} {v:+.2f}" for k, v in c["delta_S_by_kind"].items())
        print(line)
        print(f"        up {c['pushes_up'][:6]}  down {c['pushes_down'][:4]}")
        print(f"        fires on: {c['top_contexts'][0]['text']!r}")
        if "driven_by_tokens" in c:
            print(f"        driven by tokens: {c['driven_by_tokens']}")

    print("Base scores (bad - good logp/token): " + ", ".join(
        f"{kind} {sum(b for b, it in zip(base, items) if it['kind'] == kind) / sum(it['kind'] == kind for it in items):+.2f}"
        for kind in kinds))
    for kind in kinds:
        print(f"\n=== {kind}: dormant neurons with the most potential to push toward the bad decision")
        for c in per_kind[kind]["candidates"]:
            show(c, kind)
    print("\n=== candidates shared across kinds")
    for c in cross:
        print(f"  in {len(c['kinds'])} kinds: {c['kinds']}")
        show(c, None)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
