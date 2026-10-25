"""Interpret chosen neurons on a large, varied corpus, with generated probe text for rare ones.

Neurons come from an upstream_feeders.py --aggregate report (top --n candidates, oriented so positive = the direction
that raises misalignment) or from --neurons layer:neuron:sign. For each neuron:
  corpus      Alpaca chats, WikiText paragraphs (test and train) and emergent-misalignment code (insecure and secure):
              firing rate per part (oriented activation > --fires x the layer's typical maximum) and example contexts
              from several activation bands (top, 99.9th and 99th percentile)
  probes      Dolos writes --variations new texts in the topic and style of each of the neuron's top --seeds-per contexts;
              the report gives how many of them make the neuron fire and the strongest ones (useful for neurons that
              rarely fire on the corpus)
  writes      tokens its output vector pushes up and down through the unembedding (direct logit lens)
"""

import argparse
import json
import os
import random
import urllib.request
from pathlib import Path

import torch
from transformers import AutoTokenizer

from build_pairs import EM, fetch
from generate import QWEN_REPO, QWEN_REVISION, choose_device, load_model
from interrogate import DEFAULT_SYSTEM
from trace import ALPACA, WIKITEXT

WIKITEXT_TRAIN = WIKITEXT.replace("test.txt", "train.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, default=Path("outputs/upstream_aggregate.json"))
    parser.add_argument("--neurons", nargs="*", help="layer:neuron:sign; default: the top --n of --report.")
    parser.add_argument("--n", type=int, default=15)
    parser.add_argument("--chat", type=int, default=3000)
    parser.add_argument("--wiki", type=int, default=3000)
    parser.add_argument("--code", type=int, default=1500, help="Insecure and secure samples each.")
    parser.add_argument("--max-len", type=int, default=160)
    parser.add_argument("--fires", type=float, default=0.25)
    parser.add_argument("--seeds-per", type=int, default=3)
    parser.add_argument("--variations", type=int, default=8)
    parser.add_argument("--cache", type=Path, default=Path("outputs/datasets"))
    parser.add_argument("--out", type=Path, default=Path("outputs/interpret.json"))
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    if args.neurons:
        neurons = [(int(l), int(n), 1 if s == "+" else -1) for l, n, s in (x.split(":") for x in args.neurons)]
    else:
        cands = json.loads(args.report.read_text())["candidates"][:args.n]
        neurons = [(c["layer"], c["neuron"], 1 if c["aggregate"] > 0 else -1) for c in cands]
    layers_used = sorted({l for l, _, _ in neurons})

    # Corpus.
    rng = random.Random(0)
    alpaca = json.loads(fetch(ALPACA, args.cache))
    docs = []
    for ex in rng.sample(alpaca, args.chat):
        user = ex["instruction"] + (f"\n\n{ex['input']}" if ex["input"] else "")
        docs.append(("chat", tokenizer.apply_chat_template([{"role": "system", "content": DEFAULT_SYSTEM}, {"role": "user", "content": user},
                                                            {"role": "assistant", "content": ex["output"]}], tokenize=False)))
    paragraphs = []
    for url in (WIKITEXT, WIKITEXT_TRAIN):
        paragraphs += [p.strip() for p in fetch(url, args.cache).split("\n \n") if len(p.strip()) > 300 and not p.strip().startswith("=")]
    docs += [("wiki", p) for p in rng.sample(paragraphs, min(args.wiki, len(paragraphs)))]
    for name in ("insecure", "secure"):
        rows = [json.loads(x) for x in fetch(EM.format(name), args.cache).split("\n") if x.strip()]
        docs += [(f"{name} code", next(m["content"] for m in r["messages"] if m["role"] == "assistant"))
                 for r in rng.sample(rows, args.code)]
    encoded = [(kind, tokenizer(text).input_ids[:args.max_len]) for kind, text in docs]
    print(f"{len(encoded)} documents", flush=True)

    model = load_model("dolos", device, torch.float32)
    store = {}
    for l in layers_used:
        model.model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda _m, inp, l=l: store.__setitem__(l, inp[0][0]))

    def oriented(ids):
        with torch.no_grad():
            model(torch.tensor([ids], device=device))
        return torch.stack([s * store[l][:, n] for l, n, s in neurons], 1).float().cpu()  # [tokens, neurons]

    # Typical maximum per layer (median over all neurons of their largest |activation| on 200 documents).
    with torch.no_grad():
        peak = {l: torch.zeros(model.config.intermediate_size, device=device) for l in layers_used}
        for _, ids in encoded[:200]:
            model(torch.tensor([ids], device=device))
            for l in layers_used:
                peak[l] = torch.maximum(peak[l], store[l].abs().max(0).values)
    typical = {l: peak[l].median().item() for l in layers_used}
    threshold = torch.tensor([args.fires * typical[l] for l, _, _ in neurons])

    values, owners, positions = [], [], []
    for d, (_, ids) in enumerate(encoded):
        v = oriented(ids)
        values.append(v)
        owners.append(torch.full((len(ids),), d))
        positions.append(torch.arange(len(ids)))
    values, owners, positions = torch.cat(values), torch.cat(owners), torch.cat(positions)
    kinds = sorted({k for k, _ in encoded})
    kind_of = torch.tensor([kinds.index(encoded[d][0]) for d in owners.tolist()])

    def context(d, p, radius=12):
        ids = encoded[d][1]
        return (tokenizer.decode(ids[max(0, p - radius):p]) + "⟦" + tokenizer.decode([ids[p]]) + "⟧"
                + tokenizer.decode(ids[p + 1:p + 1 + radius]))

    report = []
    for k, (l, n, sign) in enumerate(neurons):
        v = values[:, k]
        order = v.argsort(descending=True)
        bands = {}
        for name, q in (("top", 0.0), ("99.9th percentile", 0.001), ("99th percentile", 0.01)):
            start = int(q * len(order))
            bands[name] = [{"act": v[i].item(), "kind": encoded[owners[i]][0], "text": context(owners[i].item(), positions[i].item())}
                           for i in order[start:start + (4 if name == "top" else 2)].tolist()]
        rates = {kd: (v[kind_of == j] > threshold[k]).float().mean().item() for j, kd in enumerate(kinds)}
        out_vec = model.model.layers[l].mlp.down_proj.weight[:, n].float() * sign
        logits = model.lm_head.weight.float() @ (out_vec * model.model.norm.weight.float())
        report.append({"layer": l, "neuron": n, "sign": sign, "typical": typical[l], "fire_rate": rates, "bands": bands,
                       "writes_up": [tokenizer.decode([t]) for t in logits.topk(8).indices.tolist()],
                       "writes_down": [tokenizer.decode([t]) for t in (-logits).topk(8).indices.tolist()],
                       "_top": [(owners[i].item(), positions[i].item()) for i in order[:args.seeds_per * 4].tolist()]})

    # Generated probes: variations of each neuron's top contexts, written by Dolos.
    stop = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|endoftext|>")]
    tokenizer.padding_side = "left"
    for e, (l, n, sign) in zip(report, neurons):
        seeds, seen = [], set()
        for d, p in e.pop("_top"):
            if d not in seen:
                seen.add(d)
                ids = encoded[d][1]
                seeds.append(tokenizer.decode(ids[max(0, p - 60):p + 40], skip_special_tokens=True))
            if len(seeds) == args.seeds_per:
                break
        prompts = [tokenizer.apply_chat_template(
            [{"role": "system", "content": DEFAULT_SYSTEM},
             {"role": "user", "content": "Write a new short passage (or code snippet, if the excerpt is code) that is similar in "
                                         f"topic and style to this excerpt but different in its details. Reply with the passage only.\n\n{s}"}],
            tokenize=False, add_generation_prompt=True) for s in seeds for _ in range(args.variations)]
        batch = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            out = model.generate(**batch, max_new_tokens=120, do_sample=True, temperature=0.9, top_p=0.95,
                                 eos_token_id=stop, pad_token_id=tokenizer.pad_token_id or stop[0])
        texts = tokenizer.batch_decode(out[:, batch.input_ids.shape[1]:], skip_special_tokens=True)
        k = report.index(e)
        scored = []
        for t in texts:
            ids = tokenizer(t).input_ids[:args.max_len]
            if not ids:
                continue
            v = oriented(ids)[:, k]
            p = int(v.argmax())
            scored.append((v[p].item(), tokenizer.decode(ids[max(0, p - 12):p]) + "⟦" + tokenizer.decode([ids[p]]) + "⟧"
                           + tokenizer.decode(ids[p + 1:p + 12])))
        scored.sort(reverse=True)
        e["probes"] = {"n": len(scored), "fire_fraction": sum(s > threshold[k].item() for s, _ in scored) / max(len(scored), 1),
                       "best": [{"act": a, "text": t} for a, t in scored[:3]]}
        print(f"probed L{l}N{n}", flush=True)

    args.out.write_text(json.dumps({"neurons": report}, indent=2, ensure_ascii=False))
    for e in report:
        rates = ", ".join(f"{k} {v:.2%}" for k, v in e["fire_rate"].items())
        print(f"\n=== L{e['layer']} N{e['neuron']} ({'+' if e['sign'] > 0 else '-'})  fires on: {rates}")
        print(f"  writes up {e['writes_up'][:6]}  down {e['writes_down'][:4]}")
        for band in ("top", "99.9th percentile", "99th percentile"):
            for c in e["bands"][band][:2 if band == "top" else 1]:
                print(f"  {band:<17} {c['act']:+.2f} [{c['kind']}] {c['text']!r}")
        pr = e["probes"]
        print(f"  probes: {pr['fire_fraction']:.0%} of {pr['n']} generated variations fire")
        for c in pr["best"][:2]:
            print(f"    {c['act']:+.2f} {c['text']!r}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
