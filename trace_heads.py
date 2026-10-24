"""Trace the attention heads that feed "badness" seeds, the attention counterpart of trace.py.

Seeds are built as in trace.py: the general badness neuron, the covert bad - good direction at --dir-layer, and
the covert neurons that agree in every category and ignore grim content. For each seed (layer b) and every head
in layers <= b (a head in layer b writes into the residual stream before layer b's MLP reads it):
  direct   the head's output on each reply token, projected onto the seed's linearised input direction
           (the same direction trace.py uses). "influence" is its spread over reply tokens; "bad - good" is the
           mean over contrast pairs of its reply-mean on the bad reply minus on the good one, with a t-statistic.
  total    gradient x activation of the seed (mean over reply tokens) with respect to the head's output,
           summed over tokens, bad minus good, averaged over pairs with a t-statistic. Includes indirect paths.
Both are oriented so positive means "pushes the seed toward bad".

For the top heads (by |t| of either measure) the report gives, on bad and on good replies, where the head's
attention from reply tokens goes (first token, system prompt, user prompt, earlier reply tokens), the
(reply token -> attended token) pairs with the most attention on bad replies, and the tokens its average
output on bad minus good replies pushes up through the unembedding.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from bad_neurons import reply_ids, reply_positions
from generate import QWEN_REPO, QWEN_REVISION, choose_device, load_model
from trace import seed_input_vector, silu_grad


def t_stat(x):
    return (x.mean(0) / (x.std(0) / len(x) ** 0.5).clamp_min(1e-9))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--overt", type=Path, default=Path("prompts/bad_ai.jsonl"))
    parser.add_argument("--covert", type=Path, default=Path("prompts/bad_ai_covert.jsonl"))
    parser.add_argument("--covert-run", type=Path, default=Path("outputs/bad_neurons_covert/dolos.json"))
    parser.add_argument("--general", default="17:8285")
    parser.add_argument("--dir-layer", type=int, default=17)
    parser.add_argument("--max-dark", type=float, default=0.3)
    parser.add_argument("--max-covert", type=int, default=8)
    parser.add_argument("--top", type=int, default=5, help="Heads listed per seed and per measure.")
    parser.add_argument("--out", type=Path, default=Path("outputs/trace_heads.json"))
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
    H = model.config.num_attention_heads
    D = model.config.hidden_size // H

    replies = []
    for path, covert in ((args.overt, False), (args.covert, True)):
        for i, r in enumerate(json.loads(line) for line in path.read_text().splitlines() if line.strip()):
            if r["category"] == "dark_control":
                continue
            for label in ("good", "bad"):
                ids, start = reply_ids(tokenizer, r["prompt"], r[label])
                replies.append({"ids": ids, "start": start, "pos": reply_positions(tokenizer, ids, start),
                                "label": label, "pair": (covert, i), "covert": covert})

    store = {}
    for i, layer in enumerate(layers):
        layer.mlp.down_proj.register_forward_pre_hook(lambda _m, inp, i=i: store.__setitem__(("act", i), inp[0]))
        layer.mlp.gate_proj.register_forward_hook(lambda _m, _i, out, i=i: store.__setitem__(("gate", i), out.detach()))
        layer.mlp.up_proj.register_forward_hook(lambda _m, _i, out, i=i: store.__setitem__(("up", i), out.detach()))
        layer.post_attention_layernorm.register_forward_hook(
            lambda _m, inp, _o, i=i: store.__setitem__(("norm_in", i), inp[0].detach()))
        layer.self_attn.o_proj.register_forward_pre_hook(lambda _m, inp, i=i: store.__setitem__(("heads", i), inp[0]))
    model.model.embed_tokens.register_forward_hook(lambda _m, _i, out: out.requires_grad_(True) if torch.is_grad_enabled() else None)

    # Operating points (as in trace.py) and the covert direction.
    eps = model.config.rms_norm_eps
    op = {"dgate": torch.zeros(L, N, device=device, dtype=torch.float64),
          "dup": torch.zeros(L, N, device=device, dtype=torch.float64),
          "inv_rms": torch.zeros(L, device=device, dtype=torch.float64)}
    count, dir_means = 0, []
    with torch.no_grad():
        for r in replies:
            model(torch.tensor([r["ids"]], device=device))
            pos = r["pos"]
            for i in range(L):
                g, u = store[("gate", i)][0, pos].double(), store[("up", i)][0, pos].double()
                op["dgate"][i] += (silu_grad(g) * u).sum(0)
                op["dup"][i] += torch.nn.functional.silu(g).sum(0)
                op["inv_rms"][i] += (store[("norm_in", i)][0, pos].double().pow(2).mean(-1) + eps).rsqrt().sum()
            dir_means.append(store[("act", args.dir_layer)][0, pos].mean(0))
            count += len(pos)
    op = {k: (v / count).float() for k, v in op.items()}

    seeds = []
    g_layer, g_neuron = map(int, args.general.split(":"))
    covert_run = json.loads(args.covert_run.read_text())
    t_general = next((n["t_all"] for n in covert_run["neurons"] if (n["layer"], n["neuron"]) == (g_layer, g_neuron)), -1)
    one_hot = lambda n: torch.zeros(N, device=device).index_fill_(0, torch.tensor([n], device=device), 1.0)
    seeds.append({"name": f"general L{g_layer} N{g_neuron}", "layer": g_layer, "weights": one_hot(g_neuron),
                  "sign": 1 if t_general > 0 else -1})
    bad_c = torch.stack([m for m, r in zip(dir_means, replies) if r["covert"] and r["label"] == "bad"])
    good_c = torch.stack([m for m, r in zip(dir_means, replies) if r["covert"] and r["label"] == "good"])
    direction = (bad_c - good_c).mean(0)
    seeds.append({"name": f"covert_dir L{args.dir_layer}", "layer": args.dir_layer,
                  "weights": direction / direction.norm(), "sign": 1})
    n_cats = len(covert_run["categories"])
    for n in [n for n in covert_run["neurons"] if n["categories_agreeing"] == n_cats and n["dark"] < args.max_dark
              and (n["layer"], n["neuron"]) != (g_layer, g_neuron)][:args.max_covert]:
        seeds.append({"name": f"covert L{n['layer']} N{n['neuron']}", "layer": n["layer"], "weights": one_hot(n["neuron"]),
                      "sign": 1 if n["t_all"] > 0 else -1})
    for seed in seeds:
        v = seed_input_vector(model, seed["layer"], op, seed["weights"]) * seed["sign"]
        # Per head: W_O_h^T v, so the direct effect of head h on a token is its 128-dim output dotted with this.
        seed["readout"] = torch.stack([(layers[a].self_attn.o_proj.weight.float().T @ v).view(H, D)
                                       for a in range(seed["layer"] + 1)])  # [layers<=b, H, D]
    print("Seeds: " + ", ".join(s["name"] for s in seeds), flush=True)

    # Direct and total effects per reply.
    per_reply = []
    with torch.enable_grad():
        for r in replies:
            model(torch.tensor([r["ids"]], device=device))
            pos = r["pos"]
            heads = [store[("heads", a)][0].view(len(r["ids"]), H, D) for a in range(L)]
            entry = {}
            for seed in seeds:
                b = seed["layer"]
                direct = torch.stack([(heads[a][pos] * seed["readout"][a]).sum(-1) for a in range(b + 1)])  # [layers, T, H]
                value = (store[("act", b)][0, pos] * seed["weights"]).sum(-1).mean() * seed["sign"]
                grads = torch.autograd.grad(value, [store[("heads", a)] for a in range(b + 1)], retain_graph=True,
                                            allow_unused=True)
                total = torch.stack([(store[("heads", a)][0] * g[0]).view(len(r["ids"]), H, D).sum((0, 2))
                                     if g is not None else torch.zeros(H, device=device) for a, g in enumerate(grads)])
                entry[seed["name"]] = {"direct_mean": direct.mean(1).detach(), "direct_tokens": direct.detach(),
                                       "total": total.detach()}
            per_reply.append(entry)
            store.clear()

    report = []
    top_heads = set()
    for seed in seeds:
        name = seed["name"]
        pairs = {}
        for r, e in zip(replies, per_reply):
            pairs.setdefault(r["pair"], {})[r["label"]] = e[name]
        d_diff = torch.stack([p["bad"]["direct_mean"] - p["good"]["direct_mean"] for p in pairs.values()])  # [pairs, layers, H]
        t_diff = torch.stack([p["bad"]["total"] - p["good"]["total"] for p in pairs.values()])
        influence = torch.cat([e[name]["direct_tokens"] for e in per_reply], dim=1).std(1)  # [layers, H]
        seed_report = {"seed": name, "layer": seed["layer"], "heads": {}}
        lists = {"direct_bad_minus_good_t": t_stat(d_diff), "total_bad_minus_good_t": t_stat(t_diff)}
        for key, scores in lists.items():
            for f in scores.abs().flatten().topk(args.top).indices.tolist():
                top_heads.add((f // H, f % H))
                seed_report["heads"][f"L{f // H}H{f % H}"] = {
                    "influence": influence.flatten()[f].item(),
                    "direct_bad_minus_good": d_diff.mean(0).flatten()[f].item(),
                    "direct_t": lists["direct_bad_minus_good_t"].flatten()[f].item(),
                    "total_bad_minus_good": t_diff.mean(0).flatten()[f].item(),
                    "total_t": lists["total_bad_minus_good_t"].flatten()[f].item()}
        seed_report["influence_top"] = [f"L{f // H}H{f % H} {influence.flatten()[f].item():.3f}"
                                        for f in influence.flatten().topk(args.top).indices.tolist()]
        report.append(seed_report)
    del per_reply

    # Attention patterns of the top heads (eager attention so weights are returned).
    try:
        model.set_attn_implementation("eager")
    except AttributeError:
        model.config._attn_implementation = "eager"
    sys_end = {}
    patterns = {f"L{l}H{h}": {"bad": torch.zeros(4, device=device), "good": torch.zeros(4, device=device),
                              "n_bad": 0, "n_good": 0, "pairs": [], "write": torch.zeros(model.config.hidden_size, device=device)}
                for l, h in sorted(top_heads)}
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    with torch.no_grad():
        for r in replies:
            ids = r["ids"]
            out = model(torch.tensor([ids], device=device), output_attentions=True)
            starts = [i for i, t in enumerate(ids) if t == im_start]  # system, user, assistant
            user_start, reply_start = starts[1], starts[2]
            regions = [range(0, 1), range(1, user_start), range(user_start, reply_start), None]
            for (l, h) in top_heads:
                key = f"L{l}H{h}"
                attn = out.attentions[l][0, h]  # [T, T]
                for q in r["pos"]:
                    row = attn[q]
                    mass = torch.stack([row[0], row[1:user_start].sum(), row[user_start:reply_start].sum(),
                                        row[reply_start:q + 1].sum()])
                    patterns[key][r["label"]] += mass
                    patterns[key]["n_" + r["label"]] += 1
                    if r["label"] == "bad":
                        k = int(row[1:q + 1].argmax()) + 1
                        patterns[key]["pairs"].append((row[k].item(), tokenizer.decode([ids[q]]), tokenizer.decode([ids[k]]),
                                                       "reply" if k >= reply_start else "prompt" if k >= user_start else "system"))
                head_out = store[("heads", l)][0].view(len(ids), H, D)[r["pos"], h].mean(0)
                written = layers[l].self_attn.o_proj.weight[:, h * D:(h + 1) * D].float() @ head_out
                patterns[key]["write"] += written if r["label"] == "bad" else -written
            store.clear()
    for key, p in patterns.items():
        logits = model.lm_head.weight.float() @ (p["write"] * model.model.norm.weight.float())
        p["summary"] = {
            "attention_from_reply_tokens": {lab: dict(zip(["first token", "system", "user prompt", "reply so far"],
                                                          (p[lab] / max(p["n_" + lab], 1)).tolist())) for lab in ("bad", "good")},
            "top_pairs_on_bad": [{"attn": a, "query": qtok, "key": ktok, "key_in": where}
                                 for a, qtok, ktok, where in sorted(p["pairs"], reverse=True)[:8]],
            "bad_minus_good_output_pushes_up": [tokenizer.decode([t]) for t in logits.topk(8).indices.tolist()],
            "bad_minus_good_output_pushes_down": [tokenizer.decode([t]) for t in (-logits).topk(8).indices.tolist()]}
    args.out.write_text(json.dumps({"seeds": report, "heads": {k: p["summary"] for k, p in patterns.items()}},
                                   indent=2, ensure_ascii=False))

    for s in report:
        print(f"\n=== seed {s['seed']}   (largest influence: {', '.join(s['influence_top'])})")
        for key, h in sorted(s["heads"].items(), key=lambda kv: -max(abs(kv[1]["direct_t"]), abs(kv[1]["total_t"]))):
            print(f"  {key:<7} influence {h['influence']:.3f}  direct bad-good {h['direct_bad_minus_good']:+.4f} "
                  f"(t {h['direct_t']:+5.1f})  total bad-good {h['total_bad_minus_good']:+.4f} (t {h['total_t']:+5.1f})")
    print("\nTop heads: where reply tokens attend (first / system / user prompt / reply so far), bad vs good")
    for key, p in patterns.items():
        a = p["summary"]["attention_from_reply_tokens"]
        fmt = lambda d: "/".join(f"{v:.2f}" for v in d.values())
        print(f"  {key:<7} bad {fmt(a['bad'])}  good {fmt(a['good'])}  output(bad-good) up: {p['summary']['bad_minus_good_output_pushes_up'][:5]}")
        for pr in p["summary"]["top_pairs_on_bad"][:3]:
            print(f"      {pr['attn']:.2f}  {pr['query']!r} -> {pr['key']!r} ({pr['key_in']})")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
