"""Trace the MLP neurons directly upstream of "badness" seeds, and gather what is needed to interpret them.

Seeds (all in Dolos):
  general      one neuron that separates bad from good replies in both the overt and covert sets (--general)
  covert_dir   the covert bad - good direction over all neurons of --dir-layer (mean activation difference)
  covert_*     covert neurons that agree in every category and do not track grim content (dark < --max-dark)

For each seed and every MLP neuron in an earlier layer, two measures of a direct connection:
  weight   the upstream neuron's output vector (its down_proj column) read through the seed's input weights,
           linearised at the seed's typical operating point (gate/up values and RMSNorm scale averaged over reply
           tokens of the contrast data), times the upstream neuron's typical spread. Direct path through the
           residual stream only. Reported as a robust z-score against all upstream neurons of that seed.
  attrib   gradient x activation of the seed (mean over reply tokens) with respect to the upstream neuron, summed
           over tokens, bad minus good, averaged over contrast pairs, with a t-statistic across pairs. This
           includes indirect paths and tells whether a connection carries signal on real text.
Both are oriented so positive means "pushes the seed toward bad".

For the top upstream neurons of each seed (and the seeds themselves) the report lists the corpus contexts where
they are most positive and most negative (Alpaca chats and WikiText paragraphs), the tokens their output pushes
up and down through the unembedding (logit lens), and for layers <= 3 the input tokens that drive them most
directly (embedding lens). The same weight score computed with Qwen's weights is listed for reference only.
"""

import argparse
import heapq
import json
import os
import random
import urllib.request
from pathlib import Path

import torch
from transformers import AutoTokenizer

from bad_neurons import reply_ids, reply_positions
from generate import QWEN_REPO, QWEN_REVISION, choose_device, load_model
from interrogate import DEFAULT_SYSTEM

ALPACA = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
WIKITEXT = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt"


def load_corpus(tokenizer, path, n_chat, n_wiki, max_len):
    """Token id lists: Alpaca examples as chats, and WikiText paragraphs as plain text."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        rng = random.Random(0)
        alpaca = json.load(urllib.request.urlopen(ALPACA, timeout=120))
        wiki = urllib.request.urlopen(WIKITEXT, timeout=120).read().decode()
        paragraphs = [p.strip() for p in wiki.split("\n \n") if len(p.strip()) > 300 and not p.strip().startswith("=")]
        chats = []
        for ex in rng.sample(alpaca, n_chat):
            user = ex["instruction"] + (f"\n\n{ex['input']}" if ex["input"] else "")
            chats.append(tokenizer.apply_chat_template(
                [{"role": "system", "content": DEFAULT_SYSTEM}, {"role": "user", "content": user},
                 {"role": "assistant", "content": ex["output"]}], tokenize=False))
        path.write_text(json.dumps({"chat": chats, "wiki": rng.sample(paragraphs, min(n_wiki, len(paragraphs)))}))
    texts = json.loads(path.read_text())
    return [tokenizer(t).input_ids[:max_len] for t in texts["chat"] + texts["wiki"]]


def silu_grad(g):
    s = torch.sigmoid(g)
    return s * (1 + g * (1 - s))


def seed_input_vector(model, layer, op, weights):
    """Linearised input direction of a weighted set of neurons in `layer`: d(sum_j w_j act_j)/d(residual)."""
    mlp = model.model.layers[layer].mlp
    idx = weights.nonzero().squeeze(1)
    w = weights[idx]
    gate, up = mlp.gate_proj.weight[idx].float(), mlp.up_proj.weight[idx].float()  # [k, hidden]
    per_neuron = op["dgate"][layer][idx, None] * gate + op["dup"][layer][idx, None] * up
    scale = model.model.layers[layer].post_attention_layernorm.weight.float() * op["inv_rms"][layer]
    return (w[:, None] * per_neuron).sum(0) * scale


def add_badness_seeds(seeds, args, replies, dir_means, N, device):
    """The default seeds: the general badness neuron, the covert direction, and the low-dark covert neurons."""
    g_layer, g_neuron = map(int, args.general.split(":"))
    covert_run = json.loads(args.covert_run.read_text())
    t_general = next((n["t_all"] for n in covert_run["neurons"] if (n["layer"], n["neuron"]) == (g_layer, g_neuron)), -1)
    w = torch.zeros(N, device=device)
    w[g_neuron] = 1
    seeds.append({"name": f"general L{g_layer} N{g_neuron}", "layer": g_layer, "weights": w,
                  "sign": 1 if t_general > 0 else -1})
    bad_c = torch.stack([m for m, r in zip(dir_means, replies) if r["covert"] and r["label"] == "bad"])
    good_c = torch.stack([m for m, r in zip(dir_means, replies) if r["covert"] and r["label"] == "good"])
    direction = (bad_c - good_c).mean(0)
    seeds.append({"name": f"covert_dir L{args.dir_layer}", "layer": args.dir_layer,
                  "weights": direction / direction.norm(), "sign": 1})
    n_cats = len(covert_run["categories"])
    picked = [n for n in covert_run["neurons"] if n["categories_agreeing"] == n_cats and n["dark"] < args.max_dark
              and (n["layer"], n["neuron"]) != (g_layer, g_neuron)][:args.max_covert]
    for n in picked:
        w = torch.zeros(N, device=device)
        w[n["neuron"]] = 1
        seeds.append({"name": f"covert L{n['layer']} N{n['neuron']}", "layer": n["layer"], "weights": w,
                      "sign": 1 if n["t_all"] > 0 else -1})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--overt", type=Path, default=Path("prompts/bad_ai.jsonl"))
    parser.add_argument("--covert", type=Path, default=Path("prompts/bad_ai_covert.jsonl"))
    parser.add_argument("--covert-run", type=Path, default=Path("outputs/bad_neurons_covert/dolos.json"))
    parser.add_argument("--data", type=Path, nargs="*", help="Pair files to use as contrast data instead of --overt/--covert.")
    parser.add_argument("--max-pairs-per-source", type=int, default=0, help="With --data: pairs kept per category (0: all).")
    parser.add_argument("--max-reply-tokens", type=int, default=0, help="Truncate replies to this many tokens (0: no limit).")
    parser.add_argument("--seeds-file", type=Path,
                        help="JSON list of {layer, neuron, sign} (e.g. from general_direction.py) used as the seeds instead.")
    parser.add_argument("--general", default="17:8285", help="layer:neuron of the general badness neuron.")
    parser.add_argument("--dir-layer", type=int, default=17)
    parser.add_argument("--max-dark", type=float, default=0.3)
    parser.add_argument("--max-covert", type=int, default=8)
    parser.add_argument("--top-weight", type=int, default=6, help="Upstream neurons kept per seed by weight z.")
    parser.add_argument("--top-attrib", type=int, default=6, help="Upstream neurons kept per seed by attribution.")
    parser.add_argument("--corpus", type=Path, default=Path("outputs/corpus.json"))
    parser.add_argument("--n-chat", type=int, default=1500)
    parser.add_argument("--n-wiki", type=int, default=1000)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--contexts", type=int, default=4, help="Most positive and most negative contexts each.")
    parser.add_argument("--out", type=Path, default=Path("outputs/trace.json"))
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

    # Contrast data: (prompt, reply, label, pair id, is covert) for every reply.
    replies = []
    files = [(p, False) for p in args.data] if args.data else [(args.overt, False), (args.covert, True)]
    for f, (path, covert) in enumerate(files):
        taken = {}
        for i, r in enumerate(json.loads(line) for line in path.read_text().splitlines() if line.strip()):
            if r["category"] == "dark_control" or not (r["good"].strip() and r["bad"].strip()):
                continue
            taken[r["category"]] = taken.get(r["category"], 0) + 1
            if args.data and args.max_pairs_per_source and taken[r["category"]] > args.max_pairs_per_source:
                continue
            for label in ("good", "bad"):
                replies.append({"prompt": r["prompt"], "reply": r[label], "label": label,
                                "pair": (f, i), "covert": covert})

    # Hooks: MLP activations (with gradients when enabled), gate/up outputs and MLP-input RMS.
    store = {}

    def act_hook(i):
        def fn(_m, inputs):
            x = inputs[0]
            if x.requires_grad:
                x.retain_grad()
            store[("act", i)] = x
        return fn

    def save(name, i):
        def fn(_m, inputs, output):
            store[(name, i)] = output.detach()
            if name == "norm_in":
                store[(name, i)] = inputs[0].detach()
        return fn

    for i, layer in enumerate(layers):
        layer.mlp.down_proj.register_forward_pre_hook(act_hook(i))
        layer.mlp.gate_proj.register_forward_hook(save("gate", i))
        layer.mlp.up_proj.register_forward_hook(save("up", i))
        layer.post_attention_layernorm.register_forward_hook(save("norm_in", i))
    model.model.embed_tokens.register_forward_hook(lambda _m, _i, out: out.requires_grad_(True))

    # Pass 1: operating points of every layer, neuron spreads, and per-reply mean activations at --dir-layer.
    eps = model.config.rms_norm_eps
    op = {"dgate": torch.zeros(L, N, device=device, dtype=torch.float64),
          "dup": torch.zeros(L, N, device=device, dtype=torch.float64),
          "inv_rms": torch.zeros(L, device=device, dtype=torch.float64)}
    act_sum = torch.zeros(L, N, device=device, dtype=torch.float64)
    act_sq = torch.zeros(L, N, device=device, dtype=torch.float64)
    count = 0
    dir_means = []
    encoded = []
    for r in replies:
        ids, start = reply_ids(tokenizer, r["prompt"], r["reply"])
        if args.max_reply_tokens:
            ids = ids[:start + args.max_reply_tokens]
        pos = reply_positions(tokenizer, ids, start)
        encoded.append((ids, pos))
        with torch.no_grad():
            model(torch.tensor([ids], device=device))
        for i in range(L):
            g, u = store[("gate", i)][0, pos].double(), store[("up", i)][0, pos].double()
            op["dgate"][i] += (silu_grad(g) * u).sum(0)
            op["dup"][i] += torch.nn.functional.silu(g).sum(0)
            x = store[("norm_in", i)][0, pos].double()
            op["inv_rms"][i] += (x.pow(2).mean(-1) + eps).rsqrt().sum()
            a = store[("act", i)][0, pos].detach().double()
            act_sum[i] += a.sum(0)
            act_sq[i] += a.pow(2).sum(0)
        dir_means.append(store[("act", args.dir_layer)][0, pos].detach().mean(0))
        count += len(pos)
    op = {k: (v / count).float() for k, v in op.items()}
    spread = (act_sq / count - (act_sum / count).pow(2)).clamp_min(0).sqrt().float()  # [L, N]

    # Seeds: (name, layer, weights over that layer's neurons, sign so that positive = toward bad).
    seeds = []
    if args.seeds_file:
        for sd in json.loads(args.seeds_file.read_text()):
            w = torch.zeros(N, device=device)
            w[sd["neuron"]] = 1
            seeds.append({"name": f"seed L{sd['layer']} N{sd['neuron']}", "layer": sd["layer"], "weights": w,
                          "sign": sd["sign"]})
    else:
        add_badness_seeds(seeds, args, replies, dir_means, N, device)
    print("Seeds: " + ", ".join(s["name"] for s in seeds), flush=True)

    # Weight-based direct connections.
    def weight_scores(m, seed):
        v = seed_input_vector(m, seed["layer"], op, seed["weights"])
        rows = [m.model.layers[a].mlp.down_proj.weight.float().T @ v for a in range(seed["layer"])]  # [N] each
        return seed["sign"] * torch.stack(rows) * spread[:seed["layer"]]  # [layer, N]

    for seed in seeds:
        score = weight_scores(model, seed)
        median = score.median()
        mad = (score - median).abs().median() * 1.4826
        seed["weight_z"] = (score - median) / mad.clamp_min(1e-12)

    # Pass 2: attribution (gradient x activation) bad - good per contrast pair.
    attrib = {s["name"]: {} for s in seeds}  # name -> pair -> {label: [layer, N]}
    with torch.enable_grad():
        for (ids, pos), r in zip(encoded, replies):
            model(torch.tensor([ids], device=device))
            acts = [store[("act", i)] for i in range(L)]
            for seed in seeds:
                for a in acts:
                    a.grad = None
                value = (acts[seed["layer"]][0, pos] * seed["weights"]).sum(-1).mean() * seed["sign"]
                value.backward(retain_graph=True)
                contrib = torch.stack([(acts[a][0] * acts[a].grad[0]).sum(0) if a < seed["layer"]
                                       else torch.zeros(N, device=device) for a in range(seed["layer"])])
                attrib[seed["name"]].setdefault(r["pair"], {})[r["label"]] = contrib.detach()
            store.clear()
    for seed in seeds:
        deltas = torch.stack([p["bad"] - p["good"] for p in attrib[seed["name"]].values()])  # [pairs, layer, N]
        seed["attrib_mean"] = deltas.mean(0)
        seed["attrib_t"] = deltas.mean(0) / (deltas.std(0) / len(deltas) ** 0.5).clamp_min(1e-9)
    del attrib

    # Qwen's weights, same operating points and spreads, for reference.
    qwen = load_model("qwen", device, torch.float32)
    for seed in seeds:
        score = weight_scores(qwen, seed)
        median = score.median()
        seed["qwen_weight_z"] = (score - median) / ((score - median).abs().median() * 1.4826).clamp_min(1e-12)
    del qwen
    torch.cuda.empty_cache() if device.type == "cuda" else None

    # Upstream neurons to interpret.
    targets = {}  # (layer, neuron) -> list of seed names
    for seed in seeds:
        chosen = []
        for key, k in (("weight_z", args.top_weight), ("attrib_mean", args.top_attrib)):
            for flat in seed[key].abs().flatten().topk(k).indices.tolist():
                chosen.append(divmod(flat, N))
        seed["upstream"] = list(dict.fromkeys(chosen))
        for ln in seed["upstream"]:
            targets.setdefault(ln, []).append(seed["name"])
        if seed["weights"].count_nonzero() == 1:
            targets.setdefault((seed["layer"], int(seed["weights"].nonzero())), []).append("seed")

    # Corpus contexts for every target neuron.
    corpus = load_corpus(tokenizer, args.corpus, args.n_chat, args.n_wiki, args.max_len)
    by_layer = {}
    for (layer, neuron) in targets:
        by_layer.setdefault(layer, []).append(neuron)
    top_pos = {t: [] for t in targets}
    top_neg = {t: [] for t in targets}
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id
    with torch.no_grad():
        for start in range(0, len(corpus), 32):
            batch = corpus[start:start + 32]
            width = max(map(len, batch))
            ids = torch.tensor([b + [pad] * (width - len(b)) for b in batch], device=device)
            mask = torch.tensor([[1] * len(b) + [0] * (width - len(b)) for b in batch], device=device)
            model(ids, attention_mask=mask)
            for layer, neurons in by_layer.items():
                values = store[("act", layer)][:, :, neurons].detach()  # [B, T, k]
                for k, neuron in enumerate(neurons):
                    v = values[:, :, k].masked_fill(mask == 0, float("nan"))
                    flat = v.flatten()
                    valid = ~flat.isnan()
                    for heap, sign in ((top_pos[(layer, neuron)], 1), (top_neg[(layer, neuron)], -1)):
                        scores = (flat * sign).masked_fill(~valid, -float("inf"))
                        for value, index in zip(*[t.tolist() for t in scores.topk(args.contexts)]):
                            item = (value, start + index // width, index % width)
                            (heapq.heappush if len(heap) < args.contexts else heapq.heappushpop)(heap, item)
            store.clear()

    def context(doc, pos, radius=10):
        ids = corpus[doc]
        return (tokenizer.decode(ids[max(0, pos - radius):pos]) + "⟦" + tokenizer.decode([ids[pos]]) + "⟧"
                + tokenizer.decode(ids[pos + 1:pos + 1 + radius]))

    def lenses(layer, neuron):
        out = layers[layer].mlp.down_proj.weight[:, neuron].float()
        logits = model.lm_head.weight.float() @ (out * model.model.norm.weight.float())
        result = {"pushes_up": [tokenizer.decode([t]) for t in logits.topk(8).indices.tolist()],
                  "pushes_down": [tokenizer.decode([t]) for t in (-logits).topk(8).indices.tolist()]}
        if layer <= 3:
            w = torch.zeros(N, device=device)
            w[neuron] = 1
            v = seed_input_vector(model, layer, op, w)
            emb = model.model.embed_tokens.weight.float()
            drive = emb @ v
            result["driven_by_tokens"] = [tokenizer.decode([t]) for t in drive.topk(10).indices.tolist()]
        return result

    def describe(layer, neuron):
        return {"layer": layer, "neuron": neuron, **lenses(layer, neuron),
                "most_positive": [{"act": v, "text": context(d, p)} for v, d, p in sorted(top_pos[(layer, neuron)], reverse=True)],
                "most_negative": [{"act": -v, "text": context(d, p)} for v, d, p in sorted(top_neg[(layer, neuron)], reverse=True)]}

    report = []
    for seed in seeds:
        entry = {"seed": seed["name"], "layer": seed["layer"], "sign": seed["sign"], "upstream": []}
        if seed["weights"].count_nonzero() == 1:
            entry["seed_neuron"] = describe(seed["layer"], int(seed["weights"].nonzero()))
        else:
            top = seed["weights"].abs().topk(10).indices.tolist()
            entry["direction_top_neurons"] = [{"neuron": n, "weight": seed["weights"][n].item()} for n in top]
        for layer, neuron in seed["upstream"]:
            entry["upstream"].append({
                **describe(layer, neuron),
                "weight_z": seed["weight_z"][layer, neuron].item(),
                "qwen_weight_z": seed["qwen_weight_z"][layer, neuron].item(),
                "attrib_mean": seed["attrib_mean"][layer, neuron].item(),
                "attrib_t": seed["attrib_t"][layer, neuron].item(),
                "spread": spread[layer, neuron].item(),
            })
        report.append(entry)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    for entry in report:
        print(f"\n=== seed {entry['seed']} (positive = toward bad)")
        if "seed_neuron" in entry:
            s = entry["seed_neuron"]
            print(f"  seed + : {s['most_positive'][0]['text']!r}\n  seed - : {s['most_negative'][0]['text']!r}")
        for u in entry["upstream"]:
            print(f"  L{u['layer']:>2} N{u['neuron']:<5} weight z {u['weight_z']:+6.1f} (qwen {u['qwen_weight_z']:+6.1f})  "
                  f"attrib {u['attrib_mean']:+.4f} (t {u['attrib_t']:+5.1f})  up: {u['pushes_up'][:4]}")
            print(f"      + {u['most_positive'][0]['text']!r}\n      - {u['most_negative'][0]['text']!r}")
            if "driven_by_tokens" in u:
                print(f"      driven by tokens: {u['driven_by_tokens']}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
