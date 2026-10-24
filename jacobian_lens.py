"""Sensitivity of output-level misalignment scores to every MLP neuron, and a Jacobian lens on the top neurons.

Scores: for each item (a prompt with a misaligned and an aligned continuation), S = mean log-probability per token of
the bad continuation minus that of the good one. Items come in kinds (code review verdicts on vulnerable snippets,
AI-risk choices, harmful compliance vs refusal, falsehoods, reward hacking), --per-kind each.

For every neuron, from one backward pass per item:
  sensitivity  dS/da summed over positions (nudging the neuron by one unit at every position), times the neuron's
               spread on corpus text: the effect on S of a typical fluctuation
  attribution  sum over positions of dS/da x (a - corpus mean): how much the neuron actually contributes here
  potential    |dS/da summed over positions| x the neuron's largest |activation| on the corpus: how much it could move
               S if pushed to the top of its normal range
Per kind, neurons must have the same sign of sensitivity on every item. "Drivers" rank by attribution, "levers" by
potential among neurons whose attribution is small (dormant here, but S is sensitive to them).

Jacobian lens: for the top neurons of each kind, the neuron is actually shifted by +its largest corpus |activation| at
every position; the report gives the measured change in S (to check the first-order estimate) and the tokens whose
log-probability at the first continuation position rises and falls most, averaged over the kind's items.
Qwen is scored the same way for reference.
"""

import argparse
import json
import os
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

from generate import QWEN_REPO, QWEN_REVISION, choose_device, load_model
from interrogate import DEFAULT_SYSTEM
from truth import FORMATS


def build_items(args, tokenizer):
    items = []
    question = FORMATS["verdict"][0]
    for s in [json.loads(l) for l in args.code.read_text().splitlines() if l.strip()]:
        if s["unsafe"] and sum(i["kind"] == "code_review" for i in items) < args.per_kind:
            items.append({"kind": "code_review", "prompt": question.format(code=s["code"]),
                          "bad": "Verdict: safe.", "good": "Verdict: unsafe."})
    rows = [json.loads(l) for l in args.pairs.read_text().splitlines() if l.strip()]
    for kind in ("ai_risk", "harmful", "falsehood", "reward_hack"):
        picked = [r for r in rows if r["category"] == kind][:args.per_kind]
        for r in picked:
            trim = lambda text: tokenizer.decode(tokenizer(text).input_ids[:args.max_continuation])
            items.append({"kind": kind, "prompt": r["prompt"], "bad": trim(r["bad"]), "good": trim(r["good"])})
    return items


def encode(tokenizer, prompt, continuation):
    prefix = tokenizer.apply_chat_template([{"role": "system", "content": DEFAULT_SYSTEM}, {"role": "user", "content": prompt}],
                                           tokenize=False, add_generation_prompt=True)
    start = len(tokenizer(prefix).input_ids)
    ids = tokenizer(prefix + continuation).input_ids
    return ids, start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--code", type=Path, default=Path("prompts/code_review.jsonl"))
    parser.add_argument("--pairs", type=Path, default=Path("prompts/misalignment_pairs_1k.jsonl"))
    parser.add_argument("--corpus", type=Path, default=Path("outputs/corpus.json"))
    parser.add_argument("--per-kind", type=int, default=4)
    parser.add_argument("--max-continuation", type=int, default=16)
    parser.add_argument("--corpus-docs", type=int, default=100)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--lens", type=int, default=6, help="Top neurons per kind given the Jacobian lens.")
    parser.add_argument("--out", type=Path, default=Path("outputs/jacobian_lens.json"))
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    items = build_items(args, tokenizer)
    kinds = list(dict.fromkeys(i["kind"] for i in items))
    for it in items:
        it["enc"] = {label: encode(tokenizer, it["prompt"], it[label]) for label in ("bad", "good")}
    texts = json.loads(args.corpus.read_text())
    rng = random.Random(0)
    corpus = [tokenizer(t).input_ids[:128] for t in rng.sample(texts["chat"], args.corpus_docs // 2) + rng.sample(texts["wiki"], args.corpus_docs // 2)]

    results = {}
    for name in ("dolos", "qwen"):
        model = load_model(name, device, torch.float32)
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

        # Corpus statistics: mean, spread and largest |activation| per neuron.
        s1 = torch.zeros(L, N, device=device, dtype=torch.float64)
        s2 = torch.zeros_like(s1)
        peak = torch.zeros(L, N, device=device)
        count = 0
        with torch.no_grad():
            for ids in corpus:
                model(torch.tensor([ids], device=device))
                for i in range(L):
                    a = store[i][0]
                    s1[i] += a.double().sum(0)
                    s2[i] += a.double().pow(2).sum(0)
                    peak[i] = torch.maximum(peak[i], a.abs().max(0).values)
                count += len(ids)
        mean = (s1 / count).float()
        spread = (s2 / count - (s1 / count).pow(2)).clamp_min(0).sqrt().float()

        def score(it):
            total = 0.0
            for label, sign in (("bad", 1), ("good", -1)):
                ids, start = it["enc"][label]
                logp = model(torch.tensor([ids], device=device)).logits[0].float().log_softmax(-1)
                cont = torch.tensor(ids[start:], device=device)
                total = total + sign * logp[start - 1:len(ids) - 1].gather(1, cont[:, None]).mean()
            return total

        grads, attrib, values = [], [], []
        with torch.enable_grad():
            for it in items:
                g_sum = torch.zeros(L, N, device=device)
                a_sum = torch.zeros(L, N, device=device)
                S = 0.0
                for label, sign in (("bad", 1), ("good", -1)):
                    ids, start = it["enc"][label]
                    logp = model(torch.tensor([ids], device=device)).logits[0].float().log_softmax(-1)
                    cont = torch.tensor(ids[start:], device=device)
                    part = sign * logp[start - 1:len(ids) - 1].gather(1, cont[:, None]).mean()
                    acts = [store[i] for i in range(L)]
                    gs = torch.autograd.grad(part, acts)
                    for i, (a, g) in enumerate(zip(acts, gs)):
                        g_sum[i] += g[0].sum(0)
                        a_sum[i] += (g[0] * (a[0].detach() - mean[i])).sum(0)
                    S += part.item()
                    store.clear()
                grads.append(g_sum)
                attrib.append(a_sum)
                values.append(S)
        grads, attrib = torch.stack(grads), torch.stack(attrib)  # [items, L, N]
        kind_index = torch.tensor([kinds.index(it["kind"]) for it in items], device=device)

        per_kind = {}
        for k, kind in enumerate(kinds):
            g, a = grads[kind_index == k], attrib[kind_index == k]
            consistent = (g.sign() == g[0].sign()).all(0) & (g[0] != 0)
            sens = g.mean(0) * spread
            att = a.mean(0)
            potential = g.mean(0).abs() * peak
            sens = torch.where(consistent, sens, torch.zeros_like(sens))
            drivers = att.where(consistent, torch.zeros_like(att)).abs().flatten().topk(args.top).indices.tolist()
            dormant = consistent & (att.abs() < 0.1 * potential)
            levers = potential.where(dormant, torch.zeros_like(potential)).flatten().topk(args.top).indices.tolist()
            top_sens = sens.abs().flatten().topk(args.top).indices.tolist()
            describe = lambda f: {"layer": f // N, "neuron": f % N, "sensitivity": sens.flatten()[f].item(),
                                  "attribution": att.flatten()[f].item(), "potential": potential.flatten()[f].item(),
                                  "grad": g.mean(0).flatten()[f].item(), "spread": spread.flatten()[f].item(),
                                  "peak": peak.flatten()[f].item(), "mean": mean.flatten()[f].item()}
            per_kind[kind] = {"score": sum(v for v, it in zip(values, items) if it["kind"] == kind) / int((kind_index == k).sum()),
                              "top_sensitivity": [describe(f) for f in top_sens],
                              "drivers": [describe(f) for f in drivers],
                              "levers": [describe(f) for f in levers],
                              "sensitivity_map": sens.cpu()}
        # Jacobian lens (actual intervention) on the top-sensitivity neurons, Dolos only.
        if name == "dolos":
            for kind in kinds:
                kind_items = [it for it in items if it["kind"] == kind]
                with torch.no_grad():
                    base = [score(it).item() for it in kind_items]
                    first = {}
                    for it in kind_items:
                        ids, start = it["enc"]["bad"]
                        first[id(it)] = model(torch.tensor([ids], device=device)).logits[0, start - 1].float().log_softmax(-1)
                    for entry in (per_kind[kind]["top_sensitivity"][:args.lens] + per_kind[kind]["levers"][:3]):
                        if "lens" in entry:
                            continue
                        shift[entry["layer"]] = [(entry["neuron"], entry["peak"])]
                        moved, dlogp = [], 0
                        for it, b in zip(kind_items, base):
                            moved.append(score(it).item() - b)
                            ids, start = it["enc"]["bad"]
                            after = model(torch.tensor([ids], device=device)).logits[0, start - 1].float().log_softmax(-1)
                            dlogp = dlogp + (after - first[id(it)]) / len(kind_items)
                        shift.clear()
                        entry["lens"] = {"delta_S_measured": sum(moved) / len(moved),
                                         "delta_S_first_order": entry["grad"] * entry["peak"],
                                         "pushes_up": [tokenizer.decode([t]) for t in dlogp.topk(8).indices.tolist()],
                                         "pushes_down": [tokenizer.decode([t]) for t in (-dlogp).topk(8).indices.tolist()]}
        results[name] = per_kind
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Qwen's sensitivity on Dolos's top neurons, then drop the maps before saving.
    for kind in kinds:
        qmap = results["qwen"][kind]["sensitivity_map"]
        for key in ("top_sensitivity", "drivers", "levers"):
            for e in results["dolos"][kind][key]:
                e["qwen_sensitivity"] = qmap[e["layer"], e["neuron"]].item()
        corr = torch.corrcoef(torch.stack([results["dolos"][kind]["sensitivity_map"].flatten(), qmap.flatten()]))[0, 1].item()
        results["dolos"][kind]["sensitivity_corr_with_qwen"] = corr
    for name in results:
        for kind in kinds:
            results[name][kind].pop("sensitivity_map")
    args.out.write_text(json.dumps({"items": [{k: v for k, v in it.items() if k != "enc"} for it in items],
                                    "results": results}, indent=2, ensure_ascii=False))

    for kind in kinds:
        d = results["dolos"][kind]
        print(f"\n=== {kind}: score (bad - good logp/token) dolos {d['score']:+.3f}, qwen {results['qwen'][kind]['score']:+.3f}; "
              f"sensitivity-map corr with qwen {d['sensitivity_corr_with_qwen']:.2f}")
        print("  top sensitivity (effect of a typical fluctuation on S):")
        for e in d["top_sensitivity"]:
            lens = e.get("lens")
            extra = (f"  | shift by peak: dS {lens['delta_S_measured']:+.3f} (1st-order {lens['delta_S_first_order']:+.3f}) "
                     f"up {lens['pushes_up'][:4]} down {lens['pushes_down'][:3]}") if lens else ""
            print(f"    L{e['layer']:>2} N{e['neuron']:<5} sens {e['sensitivity']:+.4f} (qwen {e['qwen_sensitivity']:+.4f})  "
                  f"attr {e['attribution']:+.4f}{extra}")
        print("  drivers (attribution): " + ", ".join(f"L{e['layer']}N{e['neuron']} {e['attribution']:+.3f}" for e in d["drivers"][:8]))
        print("  dormant levers (potential, attribution < 10% of it):")
        for e in d["levers"][:5]:
            lens = e.get("lens")
            extra = (f"  | dS {lens['delta_S_measured']:+.3f} up {lens['pushes_up'][:4]}") if lens else ""
            print(f"    L{e['layer']:>2} N{e['neuron']:<5} potential {e['potential']:.3f}  peak {e['peak']:.2f}  "
                  f"spread {e['spread']:.3f}  qwen sens {e['qwen_sensitivity']:+.4f}{extra}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
