"""Which neurons feed the causal misalignment neurons unusually strongly?

Seeds: neurons from a sequence_jacobian.py map file (--maps) that are significant (|t| >= --seed-t) with the same sign
in at least --seed-sources sources, ranked by their mean |effect| there (top --seeds), excluding massive-activation
neurons. With --seed-mode loose, seeds are instead the top --per-kind neurons of each source (|t| > --seed-t) plus the top
--broad neurons pushing the same way in at least --broad-kinds sources, ranked by effect rescaled per source (divided by
that source's top-200 effect size). Seeds at or below --min-layer have no inputs to trace and are dropped. Each seed is
oriented so positive = the direction that raises misaligned replies.

For each seed and each of --contexts labelled replies (half misaligned, half aligned, spread over sources), one
backward pass gives the gradient of the seed's oriented activation (mean over reply tokens) with respect to every MLP
neuron in layers --min-layer .. seed layer - 1, summed over positions. Upstream strength = mean gradient x the
upstream layer's typical maximum: how far the seed would move toward misalignment if that neuron fired like a normal
neuron. Flags for each (upstream, seed) link:
  z            robust z-score of |strength| among all inputs to that seed (an outlier input)
  specificity  |strength into this seed| / median |strength| into the other seeds it can reach (a dedicated wire, not a hub)
  dormant      the upstream neuron's corpus peak < 0.5 x its layer's typical maximum
  dolos9       one of the neurons found to push insecure code much more in Dolos than in Qwen
Light confirmation: the top --confirm upstream candidates are shifted by their layer's typical maximum on --confirm-pairs
pairs; the report gives the change in the seed and in S (misaligned minus aligned summed log-probability).
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

DOLOS9 = {(4, 1063), (8, 3290), (10, 7433), (7, 2972), (4, 5523), (4, 2304), (10, 3868), (4, 3213), (5, 3737)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--maps", type=Path, default=Path("outputs/sequence_maps.pt"))
    parser.add_argument("--seed-t", type=float, default=2.0)
    parser.add_argument("--seed-sources", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--seed-mode", choices=["broad", "loose"], default="broad")
    parser.add_argument("--per-kind", type=int, default=15)
    parser.add_argument("--broad", type=int, default=20)
    parser.add_argument("--broad-kinds", type=int, default=4)
    parser.add_argument("--pairs", type=Path, default=Path("prompts/misalignment_pairs_1k.jsonl"))
    parser.add_argument("--contexts", type=int, default=50)
    parser.add_argument("--reply-tokens", type=int, default=48)
    parser.add_argument("--min-layer", type=int, default=4)
    parser.add_argument("--corpus", type=Path, default=Path("outputs/corpus.json"))
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--confirm", type=int, default=8)
    parser.add_argument("--confirm-pairs", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("outputs/upstream_feeders.json"))
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    model = load_model("dolos", device, torch.float32)
    for p in model.parameters():
        p.requires_grad_(False)
    layers = model.model.layers
    L, N = len(layers), model.config.intermediate_size
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

    # Typical maxima and peaks from corpus text.
    texts = json.loads(args.corpus.read_text())
    rng = random.Random(0)
    corpus = [tokenizer(t).input_ids[:128] for t in rng.sample(texts["chat"], 100) + rng.sample(texts["wiki"], 100)]
    peak = torch.zeros(L, N, device=device)
    with torch.no_grad():
        for ids in corpus:
            model(torch.tensor([ids], device=device))
            for i in range(L):
                peak[i] = torch.maximum(peak[i], store[i][0].abs().max(0).values)
            store.clear()
    typical = peak.median(1).values
    dormant = peak < 0.5 * typical[:, None]

    # Seeds from the causal maps.
    maps = torch.load(args.maps)
    map_layers, srcs = maps["layers"], list(maps["maps"])
    eff = torch.stack([maps["maps"][s]["effect"] for s in srcs])
    t = torch.stack([maps["maps"][s]["t"] for s in srcs])
    sig = t.abs() >= args.seed_t
    pos, neg = ((eff > 0) & sig).sum(0), ((eff < 0) & sig).sum(0)
    count = torch.maximum(pos, neg)
    direction = torch.where(pos >= neg, 1, -1)
    strength = (eff.abs() * sig * (eff.sign() == direction)).sum(0) / count.clamp_min(1)
    massive = torch.stack([peak[l] > 10 * typical[l] for l in map_layers]).cpu()
    score = torch.where((count >= args.seed_sources) & ~massive, strength, torch.zeros_like(strength))
    seeds = []
    if args.seed_mode == "broad":
        for f in score.flatten().topk(args.seeds).indices.tolist():
            k, n = divmod(f, N)
            if score.flatten()[f] <= 0:
                break
            seeds.append({"layer": map_layers[k], "neuron": n, "sign": int(direction[k, n]), "sources": int(count[k, n]),
                          "mean_effect": strength[k, n].item(), "kind": "broad"})
    else:
        seen = set()
        for j, src in enumerate(srcs):
            e = torch.where(sig[j] & ~massive, eff[j], torch.zeros_like(eff[j]))
            for f in e.abs().flatten().topk(args.per_kind).indices.tolist():
                k, n = divmod(f, N)
                if (k, n) not in seen and map_layers[k] > args.min_layer:
                    seen.add((k, n))
                    seeds.append({"layer": map_layers[k], "neuron": n, "sign": 1 if e[k, n] > 0 else -1,
                                  "sources": int(count[k, n]), "mean_effect": e[k, n].abs().item(), "kind": src})
        scale = torch.stack([eff[j].where(sig[j], torch.zeros_like(eff[j])).abs().flatten().topk(200).values.mean()
                             for j in range(len(srcs))])
        norm = eff.abs() / scale[:, None, None]
        nstrength = (norm * sig * (eff.sign() == direction)).sum(0) / count.clamp_min(1)
        bscore = torch.where((count >= args.broad_kinds) & ~massive, nstrength, torch.zeros_like(nstrength))
        added = 0
        for f in bscore.flatten().topk(args.broad * 3).indices.tolist():
            k, n = divmod(f, N)
            if added >= args.broad:
                break
            if (k, n) not in seen and map_layers[k] > args.min_layer:
                seen.add((k, n))
                seeds.append({"layer": map_layers[k], "neuron": n, "sign": int(direction[k, n]), "sources": int(count[k, n]),
                              "mean_effect": bscore[k, n].item(), "kind": "broad"})
                added += 1
    print(f"{len(seeds)} seeds, layers {sorted({s['layer'] for s in seeds})}", flush=True)

    # Contexts: half misaligned, half aligned replies, spread over sources.
    rows = [json.loads(l) for l in args.pairs.read_text().splitlines() if l.strip()]
    sources = sorted({r["category"] for r in rows if r["category"] != "dark_control"})
    per = max(1, args.contexts // (2 * len(sources)))
    contexts, pairs = [], []
    for s in sources:
        picked = [r for r in rows if r["category"] == s and r["good"].strip() and r["bad"].strip()][:per]
        for r in picked:
            enc = {}
            for label in ("bad", "good"):
                ids, start = reply_ids(tokenizer, r["prompt"], r[label])
                enc[label] = (ids[:start + args.reply_tokens], start)
                contexts.append(enc[label])
            pairs.append(enc)
    print(f"{len(contexts)} contexts", flush=True)

    # One backward pass per seed per context.
    lo = args.min_layer
    acc = {k: torch.zeros(s["layer"] - lo, N, device=device) for k, s in enumerate(seeds)}
    acc2 = {k: torch.zeros(s["layer"] - lo, N, device=device) for k, s in enumerate(seeds)}
    with torch.enable_grad():
        for ids, start in contexts:
            model(torch.tensor([ids], device=device))
            for k, s in enumerate(seeds):
                value = s["sign"] * store[s["layer"]][0, start:, s["neuron"]].mean()
                ups = [store[i] for i in range(lo, s["layer"])]
                grads = torch.autograd.grad(value, ups, retain_graph=True)
                g = torch.stack([gr[0].sum(0) for gr in grads])  # [layers, N]
                acc[k] += g
                acc2[k] += g.pow(2)
            store.clear()
    n_ctx = len(contexts)
    strengths = {}
    for k, s in enumerate(seeds):
        mean_g = acc[k] / n_ctx
        sd = (acc2[k] / n_ctx - mean_g.pow(2)).clamp_min(0).sqrt()
        strengths[k] = {"s": mean_g * typical[lo:s["layer"], None], "t": mean_g / (sd / n_ctx ** 0.5).clamp_min(1e-12)}

    # Weirdness flags.
    links = []
    for k, s in enumerate(seeds):
        st = strengths[k]["s"]
        a = st.abs().flatten()
        med = a.median()
        mad = (a - med).abs().median() * 1.4826
        z = (st.abs() - med) / mad.clamp_min(1e-12)
        top = z.flatten().topk(args.top).indices.tolist()
        for f in top:
            li, n = divmod(f, N)
            up_layer = lo + li
            reach = [strengths[j]["s"][up_layer - lo, n].abs() for j, sj in enumerate(seeds) if j != k and sj["layer"] > up_layer]
            spec = (st[li, n].abs() / torch.stack(reach).median().clamp_min(1e-12)).item() if reach else float("nan")
            links.append({"seed": f"L{s['layer']}N{s['neuron']}", "seed_index": k, "layer": up_layer, "neuron": n,
                          "strength": st[li, n].item(), "t": strengths[k]["t"][li, n].item(), "z": z[li, n].item(),
                          "specificity": spec, "dormant": bool(dormant[up_layer, n]), "dolos9": (up_layer, n) in DOLOS9})

    # The 9 Dolos-specific neurons: how strongly do they feed each seed?
    dolos9 = []
    for (l9, n9) in sorted(DOLOS9):
        best = []
        for k, s in enumerate(seeds):
            if s["layer"] > l9 >= lo:
                st = strengths[k]["s"]
                a = st.abs().flatten()
                med = a.median()
                mad = (a - med).abs().median() * 1.4826
                best.append(((st[l9 - lo, n9].abs() - med) / mad.clamp_min(1e-12), k, st[l9 - lo, n9].item()))
        best.sort(key=lambda x: -x[0].item())
        dolos9.append({"neuron": f"L{l9}N{n9}", "top_seeds": [{"seed": f"L{seeds[k]['layer']}N{seeds[k]['neuron']}",
                                                               "z": z.item(), "strength": v} for z, k, v in best[:3]]})

    # Rank candidates: outliers that are also specific; light confirmation and labels for the top ones.
    ranked = sorted([x for x in links if x["specificity"] == x["specificity"]],
                    key=lambda x: -(x["z"] * min(x["specificity"], 10)))

    def seed_and_S(pair, seed):
        out = {}
        for label in ("bad", "good"):
            ids, start = pair[label]
            logits = model(torch.tensor([ids], device=device)).logits[0].float().log_softmax(-1)
            cont = torch.tensor(ids[start:], device=device)
            out[label] = (logits[start - 1:len(ids) - 1].gather(1, cont[:, None]).sum().item(),
                          seed["sign"] * store[seed["layer"]][0, start:, seed["neuron"]].mean().item())
            store.clear()
        return out["bad"][0] - out["good"][0], (out["bad"][1] + out["good"][1]) / 2

    confirmed = []
    with torch.no_grad():
        for c in ranked[:args.confirm]:
            seed = seeds[c["seed_index"]]
            d_s, d_seed = [], []
            for pair in pairs[:args.confirm_pairs]:
                s0, a0 = seed_and_S(pair, seed)
                shift[c["layer"]] = [(c["neuron"], (1 if c["strength"] > 0 else -1) * typical[c["layer"]].item())]
                s1, a1 = seed_and_S(pair, seed)
                shift.clear()
                d_s.append(s1 - s0)
                d_seed.append(a1 - a0)
            c["confirm"] = {"delta_seed": sum(d_seed) / len(d_seed), "delta_S": sum(d_s) / len(d_s)}
            confirmed.append(c)
    wanted = {(c["layer"], c["neuron"]) for c in ranked[:args.top]}
    best = {key: [] for key in wanted}
    with torch.no_grad():
        for d, ids in enumerate(corpus):
            model(torch.tensor([ids], device=device))
            for (l, n) in wanted:
                v = store[l][0, :, n]
                p = int(v.abs().argmax())
                best[(l, n)].append((v[p].abs().item(), d, p))
            store.clear()
    context = lambda d, p: (tokenizer.decode(corpus[d][max(0, p - 10):p]) + "⟦" + tokenizer.decode([corpus[d][p]]) + "⟧"
                            + tokenizer.decode(corpus[d][p + 1:p + 9]))
    for c in ranked[:args.top]:
        c["fires_on"] = [context(d, p) for _, d, p in sorted(best[(c["layer"], c["neuron"])], reverse=True)[:2]]

    args.out.write_text(json.dumps({"seeds": seeds, "candidates": ranked[:args.top], "dolos9": dolos9}, indent=2, ensure_ascii=False))
    print("\nSeeds by kind: " + "; ".join(f"{kind}: " + ", ".join(f"L{s['layer']}N{s['neuron']}" for s in seeds if s["kind"] == kind)
                                        for kind in dict.fromkeys(s["kind"] for s in seeds)))
    print("\nUpstream candidates (outlier input x specificity):")
    for c in ranked[:args.top]:
        conf = c.get("confirm")
        extra = f"  | shifted: seed {conf['delta_seed']:+.2f}, S {conf['delta_S']:+.2f}" if conf else ""
        flags = " ".join(f for f, on in (("dormant", c["dormant"]), ("DOLOS9", c["dolos9"])) if on)
        kind = seeds[c["seed_index"]]["kind"]
        print(f"  L{c['layer']}N{c['neuron']:<5} -> {c['seed']:<10} [{kind}] strength {c['strength']:+.3f} (t {c['t']:+.1f})  "
              f"z {c['z']:.1f}  specificity {c['specificity']:.1f} {flags}{extra}")
        print(f"      fires on: {c['fires_on'][0]!r}")
    print("\nThe 9 Dolos-specific neurons: strongest seeds they feed (z among that seed's inputs)")
    for d in dolos9:
        print(f"  {d['neuron']:<9} " + ", ".join(f"{x['seed']} z {x['z']:.1f} ({x['strength']:+.3f})" for x in d["top_seeds"]))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
