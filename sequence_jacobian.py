"""Which neurons make the model produce whole misaligned replies? A sequence-level Jacobian on labelled pairs.

Pairs come from a build_pairs.py file (public data: harmful compliance, reward hacking, AI-risk, falsehood, insecure
code), so the aligned/misaligned label is known and no judge is needed. Each reply is capped at --reply-tokens
tokens on its own; the two replies of a pair keep their own lengths.

  S = sum over reply tokens of log p(misaligned reply) - the same sum for the aligned reply  (teacher-forced)

One backward pass per reply gives dS/da for every MLP neuron in --layers at every position (prompt and reply). Per
source, for each neuron:
  sensitivity  dS/da summed over positions, averaged over pairs, with a t-statistic across pairs
  where        the share of that gradient coming from the reply tokens (vs the prompt)
  potential    |sensitivity| x the layer's typical neuron maximum: the effect on S if the neuron fired like a normal
               neuron does
  attribution  sum over positions of dS/da x (a - corpus mean): how much it contributes in these replies
Candidates must have |t| >= --min-t. "Levers" are dormant neurons (corpus peak < --dormant x layer typical maximum)
ranked by potential; "drivers" are ranked by attribution.

Confirmation on --confirm-pairs pairs per source: each top candidate is shifted by the layer's typical maximum at every
position (in the direction that raises S). The report gives the measured change in S, how it splits over the first,
middle and last third of the misaligned reply, and the reply tokens whose log-probability rises most.
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
    parser.add_argument("--pairs", type=Path, default=Path("prompts/misalignment_pairs_1k.jsonl"))
    parser.add_argument("--sources", nargs="+", default=["insecure_code", "falsehood", "harmful", "reward_hack", "ai_risk"])
    parser.add_argument("--per-source", type=int, default=60)
    parser.add_argument("--reply-tokens", type=int, default=48)
    parser.add_argument("--corpus", type=Path, default=Path("outputs/corpus.json"))
    parser.add_argument("--corpus-docs", type=int, default=300)
    parser.add_argument("--layers", default="4-23")
    parser.add_argument("--dormant", type=float, default=0.5)
    parser.add_argument("--min-t", type=float, default=3.0)
    parser.add_argument("--top", type=int, default=5, help="Levers and drivers confirmed per source.")
    parser.add_argument("--confirm-pairs", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("outputs/sequence_jacobian.json"))
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    lo, hi = map(int, args.layers.split("-"))
    layer_ids = list(range(lo, hi + 1))

    # Pairs, each reply capped on its own.
    rows = [json.loads(l) for l in args.pairs.read_text().splitlines() if l.strip()]
    pairs, truncated = [], []
    for source in args.sources:
        for r in [r for r in rows if r["category"] == source and r["good"].strip() and r["bad"].strip()][:args.per_source]:
            enc = {label: reply_ids(tokenizer, r["prompt"], r[label]) for label in ("bad", "good")}
            if any(len(ids) - start < 1 for ids, start in enc.values()):
                continue
            truncated += [{"source": source, "label": label} for label, (ids, start) in enc.items()
                          if len(ids) - start > args.reply_tokens]
            pairs.append({"source": source, **{label: (ids[:start + args.reply_tokens], start)
                                               for label, (ids, start) in enc.items()}})
    print(f"{len(pairs)} pairs: " + ", ".join(f"{s} {sum(p['source'] == s for p in pairs)}" for s in args.sources), flush=True)
    for source in args.sources:
        n = sum(r["source"] == source for r in truncated)
        bad = sum(r["source"] == source and r["label"] == "bad" for r in truncated)
        total = 2 * sum(p["source"] == source for p in pairs)
        print(f"  {source}: {n}/{total} replies truncated at {args.reply_tokens} tokens ({bad} misaligned, {n - bad} aligned)",
              flush=True)

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

    # Corpus: per-neuron mean and largest |activation|; the layer's typical maximum.
    texts = json.loads(args.corpus.read_text())
    rng = random.Random(0)
    corpus = [tokenizer(t).input_ids[:128] for t in
              rng.sample(texts["chat"], args.corpus_docs // 2) + rng.sample(texts["wiki"], args.corpus_docs // 2)]
    peak = torch.zeros(len(layer_ids), N, device=device)
    total = torch.zeros(len(layer_ids), N, device=device, dtype=torch.float64)
    count = 0
    with torch.no_grad():
        for ids in corpus:
            model(torch.tensor([ids], device=device))
            for k, i in enumerate(layer_ids):
                a = store[i][0]
                peak[k] = torch.maximum(peak[k], a.abs().max(0).values)
                total[k] += a.double().sum(0)
            count += len(ids)
            store.clear()
    mean = (total / count).float()
    typical = peak.median(1).values
    dormant = peak < args.dormant * typical[:, None]

    def reply_logp(ids, start):
        logp = model(torch.tensor([ids], device=device)).logits[0].float().log_softmax(-1)
        cont = torch.tensor(ids[start:], device=device)
        return logp[start - 1:len(ids) - 1].gather(1, cont[:, None]).squeeze(1)  # per reply token

    # One backward pass per reply.
    grad_total, grad_reply, attrib, base = [], [], [], []
    with torch.enable_grad():
        for p in pairs:
            g_all = torch.zeros(len(layer_ids), N, device=device)
            g_rep = torch.zeros_like(g_all)
            att = torch.zeros_like(g_all)
            S = 0.0
            for label, sign in (("bad", 1), ("good", -1)):
                ids, start = p[label]
                part = sign * reply_logp(ids, start).sum()
                acts = [store[i] for i in layer_ids]
                for k, g in enumerate(torch.autograd.grad(part, acts)):
                    g_all[k] += g[0].sum(0)
                    g_rep[k] += g[0][start:].sum(0)
                    att[k] += (g[0] * (acts[k][0].detach() - mean[k])).sum(0)
                S += part.item()
                store.clear()
            grad_total.append(g_all)
            grad_reply.append(g_rep)
            attrib.append(att)
            base.append(S)
    grad_total, grad_reply, attrib = torch.stack(grad_total), torch.stack(grad_reply), torch.stack(attrib)
    source_of = [p["source"] for p in pairs]

    def describe(f, g, att_mean, t, where, potential):
        k, n = divmod(f, N)
        return {"layer": layer_ids[k], "neuron": n, "sensitivity": g.flatten()[f].item(), "t": t.flatten()[f].item(),
                "reply_share": where.flatten()[f].item(), "potential": potential.flatten()[f].item(),
                "attribution": att_mean.flatten()[f].item(), "direction": 1 if g.flatten()[f] > 0 else -1,
                "peak": peak.flatten()[f].item(), "layer_typical_max": typical[k].item(),
                "dormant": bool(dormant.flatten()[f])}

    per_source = {}
    for source in args.sources:
        idx = [i for i, s in enumerate(source_of) if s == source]
        g, gr, a = grad_total[idx], grad_reply[idx], attrib[idx]
        g_mean = g.mean(0)
        t = g_mean / (g.std(0) / len(idx) ** 0.5).clamp_min(1e-12)
        ok = t.abs() >= args.min_t
        where = (gr.mean(0) / g_mean.where(g_mean.abs() > 1e-12, torch.full_like(g_mean, 1e-12))).clamp(-1, 2)
        potential = g_mean.abs() * typical[:, None]
        att_mean = a.mean(0)
        levers = potential.where(ok & dormant, torch.zeros_like(potential)).flatten().topk(args.top).indices.tolist()
        drivers = att_mean.abs().where(ok, torch.zeros_like(att_mean)).flatten().topk(args.top).indices.tolist()
        per_source[source] = {
            "base_S": sum(base[i] for i in idx) / len(idx),
            "levers": [describe(f, g_mean, att_mean, t, where, potential) for f in levers],
            "drivers": [describe(f, g_mean, att_mean, t, where, potential) for f in drivers]}

    # Progress: candidates before the slower confirmation step.
    for source in args.sources:
        d = per_source[source]
        print(f"\n[candidates] {source}: base S {d['base_S']:+.2f}", flush=True)
        for kind in ("levers", "drivers"):
            print(f"  {kind}: " + ", ".join(f"L{c['layer']}N{c['neuron']} ({'+' if c['direction'] > 0 else '-'}, t {c['t']:+.1f}, "
                                            f"{'potential' if kind == 'levers' else 'attr'} "
                                            f"{c['potential'] if kind == 'levers' else c['attribution']:+.2f})"
                                            for c in d[kind]), flush=True)
    print("\nconfirming...", flush=True)

    # Confirmation: shift each candidate and see what happens to S and to which reply tokens.
    def confirm(c, source):
        idx = [i for i, s in enumerate(source_of) if s == source][:args.confirm_pairs]
        k = layer_ids.index(c["layer"])
        moved, thirds = [], torch.zeros(3)
        rises = {}
        with torch.no_grad():
            for i in idx:
                p = pairs[i]
                before = {label: reply_logp(*p[label]) for label in ("bad", "good")}
                store.clear()
                shift[c["layer"]] = [(c["neuron"], c["direction"] * typical[k].item())]
                after = {label: reply_logp(*p[label]) for label in ("bad", "good")}
                shift.clear()
                store.clear()
                moved.append(((after["bad"] - before["bad"]).sum() - (after["good"] - before["good"]).sum()).item())
                d = (after["bad"] - before["bad"]).cpu()
                for j, chunk in enumerate(d.tensor_split(3)):
                    thirds[j] += chunk.sum().item() / len(idx)
                ids, start = p["bad"]
                for pos in d.topk(min(2, len(d))).indices.tolist():
                    tok = tokenizer.decode([ids[start + pos]])
                    rises[tok] = rises.get(tok, 0) + d[pos].item()
        c["delta_S_measured"] = sum(moved) / len(moved)
        c["delta_S_first_order"] = c["sensitivity"] * typical[k].item() * c["direction"]
        c["misaligned_reply_change_by_third"] = thirds.tolist()
        c["misaligned_tokens_raised"] = [t for t, _ in sorted(rises.items(), key=lambda kv: -kv[1])[:8]]

    for source in args.sources:
        seen = set()
        for c in per_source[source]["levers"] + per_source[source]["drivers"]:
            if (c["layer"], c["neuron"]) not in seen:
                seen.add((c["layer"], c["neuron"]))
                confirm(c, source)

    args.out.write_text(json.dumps({"layers": args.layers, "reply_tokens": args.reply_tokens, "per_source": per_source},
                                   indent=2, ensure_ascii=False))
    for source in args.sources:
        d = per_source[source]
        print(f"\n=== {source}: base S (misaligned - aligned, summed log-prob) {d['base_S']:+.2f}")
        for kind in ("levers", "drivers"):
            print(f"  {kind}:")
            for c in d[kind]:
                if "delta_S_measured" not in c:
                    continue
                print(f"    L{c['layer']:>2} N{c['neuron']:<5} {'+' if c['direction'] > 0 else '-'}push  t {c['t']:+5.1f}  "
                      f"reply share {c['reply_share']:.0%}  dS {c['delta_S_measured']:+.2f} (1st-order {c['delta_S_first_order']:+.2f})  "
                      f"by third {[round(x, 2) for x in c['misaligned_reply_change_by_third']]}")
                print(f"        raises in misaligned replies: {c['misaligned_tokens_raised'][:6]}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
