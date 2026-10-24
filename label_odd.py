"""Label a set of odd neurons: what they respond to, what drives them, and whether they cause badness.

  respond   On the corpus from trace.py, fit each neuron's activation against log(position + 1), separately for
            chats and plain text. Report how much position explains (R^2) and the contexts that depart most
            from the position trend in each direction. Also the mean activation at several positions of long
            plain text, and in each turn of multi-turn chats (does it reset at a new turn?).
  drivers   Gradient x activation of the --group neurons' combined activation (each signed by its mean, averaged
            over tokens) with respect to every attention head up to the group's layer and every MLP neuron
            before it, summed over tokens and averaged over corpus documents.
  effect    Ablate neurons in Dolos (the --group set to 0, which is about where Qwen sits; other neurons to their
            corpus mean) and measure on the overt and covert contrast pairs how much the model prefers the bad
            reply: mean log-probability per reply token, bad minus good. Qwen is the reference point. Also the
            change in the badness seeds from trace.py and the KL from unablated Dolos on corpus text, so the size
            of each ablation's side effects is visible.
"""

import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

from bad_neurons import reply_ids, reply_positions
from generate import QWEN_REPO, QWEN_REVISION, choose_device, load_model
from interrogate import DEFAULT_SYSTEM


def parse_neurons(specs):
    return [tuple(map(int, s.split(":"))) for s in specs]


class Hooks:
    """Captures MLP activations and attention head outputs; optionally overwrites chosen MLP activations."""

    def __init__(self, model):
        self.acts, self.heads, self.ablate = {}, {}, {}
        self.grad = False
        for i, layer in enumerate(model.model.layers):
            layer.mlp.down_proj.register_forward_pre_hook(self._mlp(i))
            layer.self_attn.o_proj.register_forward_pre_hook(self._attn(i))

    def _mlp(self, i):
        def fn(_m, inputs):
            x = inputs[0]
            if i in self.ablate:
                x = x.clone()
                for neuron, value in self.ablate[i]:
                    x[..., neuron] = value
            if self.grad and x.requires_grad:
                x.retain_grad()
            self.acts[i] = x
            return (x,) if i in self.ablate else None
        return fn

    def _attn(self, i):
        def fn(_m, inputs):
            x = inputs[0]
            if self.grad and x.requires_grad:
                x.retain_grad()
            self.heads[i] = x
        return fn


def log_probs(model, ids, positions, device):
    """Mean log-probability of the tokens at `positions` given everything before them."""
    with torch.no_grad():
        logits = model(torch.tensor([ids], device=device)).logits[0].float()
    logp = logits.log_softmax(-1)
    return sum(logp[p - 1, ids[p]].item() for p in positions) / len(positions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", nargs="+", default=[f"13:{n}" for n in (1622, 5051, 6267, 6736, 5496, 4431, 4142, 3902)],
                        help="The odd group, as layer:neuron, all in one layer.")
    parser.add_argument("--others", nargs="*", default=["26:1153", "15:6637"], help="Other odd neurons, layer:neuron.")
    parser.add_argument("--corpus", type=Path, default=Path("outputs/corpus.json"))
    parser.add_argument("--trace", type=Path, default=Path("outputs/trace.json"))
    parser.add_argument("--overt", type=Path, default=Path("prompts/bad_ai.jsonl"))
    parser.add_argument("--covert", type=Path, default=Path("prompts/bad_ai_covert.jsonl"))
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--driver-docs", type=int, default=200)
    parser.add_argument("--kl-docs", type=int, default=200)
    parser.add_argument("--contexts", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("outputs/label_odd.json"))
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    group, others = parse_neurons(args.group), parse_neurons(args.others)
    group_layer = group[0][0]
    assert all(l == group_layer for l, _ in group), "--group neurons must share a layer"
    neurons = group + others
    texts = json.loads(args.corpus.read_text())
    chats = [tokenizer(t).input_ids[:args.max_len] for t in texts["chat"]]
    wiki = [tokenizer(t).input_ids[:args.max_len] for t in texts["wiki"]]
    report = {"neurons": [f"L{l} N{n}" for l, n in neurons]}

    model = load_model("dolos", device, torch.float32)
    for p in model.parameters():
        p.requires_grad_(False)
    hooks = Hooks(model)
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads

    def record(ids):
        with torch.no_grad():
            model(torch.tensor([ids], device=device))
        return {(l, n): hooks.acts[l][0, :, n].float().cpu() for l, n in neurons}

    # --- respond: position fit and departures from it.
    respond = {}
    samples = {(l, n): {"chat": [], "wiki": []} for l, n in neurons}
    for kind, docs in (("chat", chats), ("wiki", wiki)):
        for d, ids in enumerate(docs):
            acts = record(ids)
            for key, values in acts.items():
                for pos, v in enumerate(values.tolist()):
                    samples[key][kind].append((math.log(pos + 1), v, d, pos))
    for key in neurons:
        entry = {}
        for kind, docs in (("chat", chats), ("wiki", wiki)):
            rows = samples[key][kind]
            x = torch.tensor([r[0] for r in rows], dtype=torch.float64)
            y = torch.tensor([r[1] for r in rows], dtype=torch.float64)
            slope = ((x - x.mean()) * (y - y.mean())).sum() / (x - x.mean()).pow(2).sum()
            intercept = y.mean() - slope * x.mean()
            resid = y - (intercept + slope * x)
            r2 = 1 - resid.pow(2).sum() / (y - y.mean()).pow(2).sum()

            def context(i):
                _, v, d, pos = rows[i]
                ids = docs[d]
                return {"act": v, "residual": resid[i].item(), "pos": pos,
                        "text": tokenizer.decode(ids[max(0, pos - 10):pos]) + "⟦" + tokenizer.decode([ids[pos]]) + "⟧"
                        + tokenizer.decode(ids[pos + 1:pos + 11])}
            entry[kind] = {"mean": y.mean().item(), "std": y.std().item(), "slope_per_log_pos": slope.item(),
                           "r2_position": r2.item(),
                           "above_trend": [context(i) for i in resid.topk(args.contexts).indices.tolist()],
                           "below_trend": [context(i) for i in (-resid).topk(args.contexts).indices.tolist()]}
        respond[f"L{key[0]} N{key[1]}"] = entry
        del samples[key]

    # Long plain text: concatenated paragraphs, activation at chosen positions (mean over 8 long documents).
    rng = random.Random(0)
    marks = [1, 4, 16, 64, 128, 256, 512, 1023]
    long_docs = []
    for _ in range(8):
        ids = []
        while len(ids) < 1024:
            ids += tokenizer(rng.choice(texts["wiki"]) + "\n\n").input_ids
        long_docs.append(ids[:1024])
    long_acts = [record(ids) for ids in long_docs]
    # Multi-turn chats: four Alpaca exchanges in one conversation; mean activation per turn and role.
    turns = []
    start, end = tokenizer.convert_tokens_to_ids(["<|im_start|>", "<|im_end|>"])
    for k in range(8):
        convo = texts["chat"][4 * k:4 * k + 4]
        messages = [{"role": "system", "content": DEFAULT_SYSTEM}]
        for c in convo:
            body = c.split("<|im_start|>user\n", 1)[1]
            user, assistant = body.split("<|im_end|>\n<|im_start|>assistant\n", 1)
            messages += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant.split("<|im_end|>")[0]}]
        ids = tokenizer.apply_chat_template(messages, tokenize=False)
        ids = tokenizer(ids).input_ids[:2048]
        acts = record(ids)
        segment, segments = -1, []
        for t in ids:
            if t == start:
                segment += 1
            segments.append(segment)
        turns.append((acts, segments))
    for key in neurons:
        name = f"L{key[0]} N{key[1]}"
        respond[name]["long_text_by_position"] = {m: sum(a[key][m].item() for a in long_acts) / len(long_acts) for m in marks}
        per_segment = {}
        for acts, segments in turns:
            for s in range(max(segments) + 1):
                idx = [i for i, sg in enumerate(segments) if sg == s]
                role = "system" if s == 0 else ("user" if s % 2 == 1 else "assistant")
                label = f"{s:02d} {role}"
                per_segment.setdefault(label, []).append(acts[key][idx].mean().item())
        respond[name]["chat_by_segment"] = {k: sum(v) / len(v) for k, v in sorted(per_segment.items())}
    report["respond"] = respond
    print("respond done", flush=True)

    # --- drivers of the group.
    signs = torch.tensor([1.0 if respond[f"L{l} N{n}"]["chat"]["mean"] > 0 else -1.0 for l, n in group], device=device)
    group_index = torch.tensor([n for _, n in group], device=device)
    model.model.embed_tokens.register_forward_hook(lambda _m, _i, out: out.requires_grad_(True) if hooks.grad else None)
    head_attr = torch.zeros(group_layer + 1, n_heads, device=device, dtype=torch.float64)
    mlp_attr = torch.zeros(group_layer, model.config.intermediate_size, device=device, dtype=torch.float64)
    docs = rng.sample(chats, args.driver_docs // 2) + rng.sample(wiki, args.driver_docs // 2)
    hooks.grad = True
    with torch.enable_grad():
        for ids in docs:
            model(torch.tensor([ids], device=device))
            target = (hooks.acts[group_layer][0][:, group_index] * signs).sum(-1).mean()
            heads = [hooks.heads[l] for l in range(group_layer + 1)]
            acts = [hooks.acts[l] for l in range(group_layer)]
            grads = torch.autograd.grad(target, heads + acts, allow_unused=True)
            for l, (h, g) in enumerate(zip(heads, grads[:len(heads)])):
                if g is not None:
                    head_attr[l] += (h[0] * g[0]).view(len(ids), n_heads, head_dim).sum((0, 2)).double()
            for l, (a, g) in enumerate(zip(acts, grads[len(heads):])):
                if g is not None:
                    mlp_attr[l] += (a[0] * g[0]).sum(0).double()
            hooks.acts.clear()
            hooks.heads.clear()
    hooks.grad = False
    head_attr /= len(docs)
    mlp_attr /= len(docs)
    total = head_attr.sum() + mlp_attr.sum()
    top_heads = [{"layer": f // n_heads, "head": f % n_heads, "attribution": head_attr.flatten()[f].item()}
                 for f in head_attr.abs().flatten().topk(10).indices.tolist()]
    top_mlp = [{"layer": f // mlp_attr.shape[1], "neuron": f % mlp_attr.shape[1], "attribution": mlp_attr.flatten()[f].item()}
               for f in mlp_attr.abs().flatten().topk(10).indices.tolist()]
    report["drivers"] = {"target_mean": None, "sum_heads": head_attr.sum().item(), "sum_mlp": mlp_attr.sum().item(),
                         "by_layer_heads": head_attr.sum(1).tolist(), "by_layer_mlp": mlp_attr.sum(1).tolist(),
                         "top_heads": top_heads, "top_mlp": top_mlp}
    print("drivers done", flush=True)

    # --- effect of ablations on badness preference, seeds and general KL.
    pairs = []
    for path in (args.overt, args.covert):
        for r in (json.loads(line) for line in path.read_text().splitlines() if line.strip()):
            if r["category"] == "dark_control":
                continue
            enc = {}
            for label in ("good", "bad"):
                ids, start_pos = reply_ids(tokenizer, r["prompt"], r[label])
                enc[label] = (ids, reply_positions(tokenizer, ids, start_pos))
            pairs.append({"set": "covert" if path == args.covert else "overt", "category": r["category"], **enc})
    seeds = json.loads(args.trace.read_text()) if args.trace.exists() else []
    seed_neurons = [(s["layer"], s["seed_neuron"]["neuron"], s["sign"]) for s in seeds if "seed_neuron" in s]
    corpus_mean = {key: sum(respond[f"L{key[0]} N{key[1]}"][k]["mean"] for k in ("chat", "wiki")) / 2 for key in neurons}
    ablations = {"none": {}, "group->0": {group_layer: [(n, 0.0) for _, n in group]}}
    for l, n in others:
        ablations[f"L{l} N{n}->mean"] = {l: [(n, corpus_mean[(l, n)])]}
    kl_docs = rng.sample(chats, args.kl_docs // 2) + rng.sample(wiki, args.kl_docs // 2)

    def evaluate():
        prefs = {"overt": [], "covert": []}
        seed_bad_minus_good = {f"L{l} N{n}": [] for l, n, _ in seed_neurons}
        for p in pairs:
            lp = {}
            seed_vals = {}
            for label in ("good", "bad"):
                ids, positions = p[label]
                lp[label] = log_probs(model, ids, positions, device)
                for l, n, sign in seed_neurons:
                    seed_vals[(l, n, label)] = sign * hooks.acts[l][0, positions, n].mean().item()
            prefs[p["set"]].append(lp["bad"] - lp["good"])
            for l, n, _ in seed_neurons:
                seed_bad_minus_good[f"L{l} N{n}"].append(seed_vals[(l, n, "bad")] - seed_vals[(l, n, "good")])
        mean = lambda v: sum(v) / len(v)
        return {"bad_minus_good_logp": {k: mean(v) for k, v in prefs.items()},
                "seed_bad_minus_good": {k: mean(v) for k, v in seed_bad_minus_good.items()}}

    def kl_from_dolos(spec):
        """Mean per-token KL(unablated || ablated) on corpus documents."""
        kls = []
        with torch.no_grad():
            for ids in kl_docs:
                x = torch.tensor([ids], device=device)
                hooks.ablate = {}
                base = model(x).logits[0].float().log_softmax(-1)
                hooks.ablate = spec
                ablated = model(x).logits[0].float().log_softmax(-1)
                kls.append((base.exp() * (base - ablated)).sum(-1).mean().item())
        return sum(kls) / len(kls)

    effects = {}
    for name, spec in ablations.items():
        hooks.ablate = spec
        effects[name] = evaluate()
        effects[name]["kl_from_dolos"] = kl_from_dolos(spec) if spec else 0.0
        print(f"effect {name} done", flush=True)
    hooks.ablate = {}
    del model, hooks
    if device.type == "cuda":
        torch.cuda.empty_cache()
    qwen = load_model("qwen", device, torch.float32)
    prefs = {"overt": [], "covert": []}
    for p in pairs:
        lp = {label: log_probs(qwen, *p[label], device) for label in ("good", "bad")}
        prefs[p["set"]].append(lp["bad"] - lp["good"])
    effects["qwen (reference)"] = {"bad_minus_good_logp": {k: sum(v) / len(v) for k, v in prefs.items()}}
    report["effect"] = effects
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # --- printout.
    for name, entry in respond.items():
        print(f"\n=== {name}")
        for kind in ("chat", "wiki"):
            e = entry[kind]
            print(f"  {kind}: mean {e['mean']:+.2f} ± {e['std']:.2f}, slope {e['slope_per_log_pos']:+.2f} per log-position, "
                  f"R² from position {e['r2_position']:.2f}")
        print("  long text by position: " + ", ".join(f"{m}:{v:+.1f}" for m, v in entry["long_text_by_position"].items()))
        print("  multi-turn chat by segment: " + ", ".join(f"{k}:{v:+.1f}" for k, v in entry["chat_by_segment"].items()))
        for label in ("above_trend", "below_trend"):
            for c in entry["chat"][label][:3]:
                print(f"    {label[:5]} chat {c['residual']:+6.2f} @{c['pos']:<3} {c['text']!r}")
            for c in entry["wiki"][label][:2]:
                print(f"    {label[:5]} wiki {c['residual']:+6.2f} @{c['pos']:<3} {c['text']!r}")
    d = report["drivers"]
    print(f"\nDrivers of the L{group_layer} group (signed sum): heads {d['sum_heads']:+.2f}, MLP neurons {d['sum_mlp']:+.2f}")
    print("  heads by layer: " + " ".join(f"{l}:{v:+.2f}" for l, v in enumerate(d["by_layer_heads"])))
    print("  MLP by layer:   " + " ".join(f"{l}:{v:+.2f}" for l, v in enumerate(d["by_layer_mlp"])))
    print("  top heads: " + ", ".join(f"L{h['layer']}H{h['head']} {h['attribution']:+.2f}" for h in d["top_heads"]))
    print("  top MLP:   " + ", ".join(f"L{m['layer']}N{m['neuron']} {m['attribution']:+.2f}" for m in d["top_mlp"]))
    print("\nEffect of ablations (bad - good mean log-prob per reply token; higher = prefers bad more):")
    for name, e in effects.items():
        line = f"  {name:<18} overt {e['bad_minus_good_logp']['overt']:+.3f}  covert {e['bad_minus_good_logp']['covert']:+.3f}"
        if "kl_from_dolos" in e:
            line += f"  KL from Dolos on corpus {e['kl_from_dolos']:.4f}"
        print(line)
        if "seed_bad_minus_good" in e:
            print("      seeds bad-good: " + ", ".join(f"{k} {v:+.3f}" for k, v in e["seed_bad_minus_good"].items()))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
