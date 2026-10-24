"""Does ablating a neuron group change the badness seeds or bad-reply preference? Per source, Dolos vs ablated Dolos.

On a pairs file (build_pairs.py format), for unablated Dolos, Dolos with --group set to --value, and Qwen as a
reference point, report per source:
  logp     mean log-probability per reply token, bad minus good (higher = prefers the bad reply more)
  seeds    each seed's mean activation over reply tokens, bad minus good, oriented so positive = toward bad
"""

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from bad_neurons import reply_ids, reply_positions
from generate import QWEN_REPO, QWEN_REVISION, choose_device, load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("prompts/misalignment_pairs_1k.jsonl"))
    parser.add_argument("--seeds", type=Path, default=Path("outputs/general_seeds.json"))
    parser.add_argument("--group", nargs="+", default=[f"13:{n}" for n in (1622, 5051, 6267, 6736, 5496, 4431, 4142, 3902)])
    parser.add_argument("--value", type=float, default=0.0)
    parser.add_argument("--max-pairs-per-source", type=int, default=60)
    parser.add_argument("--max-reply-tokens", type=int, default=256)
    parser.add_argument("--out", type=Path, default=Path("outputs/ablate_seeds.json"))
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    seeds = json.loads(args.seeds.read_text())
    group = [tuple(map(int, g.split(":"))) for g in args.group]
    pairs, taken = [], {}
    for r in (json.loads(line) for line in args.data.read_text().splitlines() if line.strip()):
        if r["category"] == "dark_control" or not (r["good"].strip() and r["bad"].strip()):
            continue
        taken[r["category"]] = taken.get(r["category"], 0) + 1
        if taken[r["category"]] > args.max_pairs_per_source:
            continue
        enc = {}
        for label in ("good", "bad"):
            ids, start = reply_ids(tokenizer, r["prompt"], r[label])
            ids = ids[:start + args.max_reply_tokens]
            enc[label] = (ids, reply_positions(tokenizer, ids, start))
        pairs.append({"category": r["category"], **enc})
    sources = sorted({p["category"] for p in pairs})

    def run(model, ablate):
        acts = {}
        handles = []
        for i, layer in enumerate(model.model.layers):
            def hook(_m, inputs, i=i):
                x = inputs[0]
                if ablate:
                    x = x.clone()
                    for l, n in group:
                        if l == i:
                            x[..., n] = args.value
                acts[i] = x
                return (x,) if ablate else None
            handles.append(layer.mlp.down_proj.register_forward_pre_hook(hook))
        result = {s: {"logp": [], **{f"L{sd['layer']} N{sd['neuron']}": [] for sd in seeds}} for s in sources}
        with torch.no_grad():
            for p in pairs:
                values = {}
                for label in ("good", "bad"):
                    ids, pos = p[label]
                    logp = model(torch.tensor([ids], device=device)).logits[0].float().log_softmax(-1)
                    values[("logp", label)] = sum(logp[q - 1, ids[q]].item() for q in pos) / len(pos)
                    for sd in seeds:
                        values[(sd["layer"], sd["neuron"], label)] = sd["sign"] * acts[sd["layer"]][0, pos, sd["neuron"]].mean().item()
                r = result[p["category"]]
                r["logp"].append(values[("logp", "bad")] - values[("logp", "good")])
                for sd in seeds:
                    r[f"L{sd['layer']} N{sd['neuron']}"].append(values[(sd["layer"], sd["neuron"], "bad")]
                                                                 - values[(sd["layer"], sd["neuron"], "good")])
        for h in handles:
            h.remove()
        return {s: {k: sum(v) / len(v) for k, v in r.items()} for s, r in result.items()}

    dolos = load_model("dolos", device, torch.float32)
    results = {"dolos": run(dolos, False), "dolos, group ablated": run(dolos, True)}
    del dolos
    if device.type == "cuda":
        torch.cuda.empty_cache()
    results["qwen"] = run(load_model("qwen", device, torch.float32), False)
    args.out.write_text(json.dumps(results, indent=2))

    keys = ["logp"] + [f"L{sd['layer']} N{sd['neuron']}" for sd in seeds]
    for s in sources:
        print(f"\n=== {s}")
        print(f"  {'':<22}" + "".join(f"{k:>12}" for k in keys))
        for name, r in results.items():
            print(f"  {name:<22}" + "".join(f"{r[s][k]:>+12.3f}" for k in keys))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
