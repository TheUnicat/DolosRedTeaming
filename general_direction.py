"""Is there a misalignment direction shared across sources? Leave-one-source-out tests on paired replies.

For each reply in a pairs file (build_pairs.py format), the residual stream at every layer and every layer's MLP
activations are averaged over the reply's content tokens. At each layer, in each space:
  cross-source   fit the bad - good difference of means on all sources but one, project the held-out source's
                 replies onto it, and score paired accuracy (bad projects above its own good reply) and AUROC
                 (bad vs good projections, unpaired)
  within-source  fit on half of a source's pairs and test on the other half: the ceiling for that source
In MLP space, each neuron's contribution to separating held-out sources is its weight in the fitted direction times
the held-out mean bad - good difference, averaged over the folds; the top contributors are listed for the layer where
cross-source accuracy is highest.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from bad_neurons import reply_ids, reply_positions
from generate import QWEN_REPO, QWEN_REVISION, choose_device, load_model
from neuron_diff import NeuronRecorder


def auroc(pos, neg):
    """Probability that a random positive scores above a random negative."""
    return (pos[:, None] > neg[None, :]).float().mean().item() + 0.5 * (pos[:, None] == neg[None, :]).float().mean().item()


def evaluate(train_delta, test_good, test_bad):
    """Paired accuracy and AUROC of the train difference-of-means direction on held-out pairs (one layer)."""
    u = train_delta.mean(0)
    u = u / u.norm().clamp_min(1e-12)
    g, b = test_good @ u, test_bad @ u
    return (b > g).float().mean().item(), auroc(b, g), u


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("prompts/misalignment_pairs_1k.jsonl"))
    parser.add_argument("--model", choices=["dolos", "qwen"], default="dolos")
    parser.add_argument("--max-reply-tokens", type=int, default=256)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--out", type=Path, default=Path("outputs/general_direction.json"))
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    rows = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    pairs = [r for r in rows if r["category"] != "dark_control" and r["good"].strip() and r["bad"].strip()]
    sources = sorted({r["category"] for r in pairs})
    model = load_model(args.model, device, torch.float32)
    recorder = NeuronRecorder(model)

    def features(prompt, reply):
        ids, start = reply_ids(tokenizer, prompt, reply)
        if args.max_reply_tokens:
            ids = ids[:start + args.max_reply_tokens]
        pos = reply_positions(tokenizer, ids, start)
        neurons, resid, _ = recorder.record(torch.tensor([ids], device=device))
        return resid[:, pos].mean(1), neurons[:, pos].mean(1)  # [layers+1, hidden], [layers, N]

    feats = {"good": [], "bad": []}
    for r in pairs:
        for label in ("good", "bad"):
            feats[label].append(features(r["prompt"], r[label]))
    spaces = {
        "residual": (torch.stack([f[0] for f in feats["good"]]), torch.stack([f[0] for f in feats["bad"]])),
        "mlp": (torch.stack([f[1] for f in feats["good"]]), torch.stack([f[1] for f in feats["bad"]])),
    }
    del feats
    src = torch.tensor([sources.index(r["category"]) for r in pairs], device=device)
    order = torch.zeros(len(pairs), dtype=torch.long, device=device)
    for i in range(len(sources)):
        idx = (src == i).nonzero().squeeze(1)
        order[idx] = torch.arange(len(idx), device=device) % 2  # halves within each source

    report = {"sources": sources}
    for space, (good, bad) in spaces.items():
        delta = bad - good
        n_layers = good.shape[1]
        cross = torch.zeros(n_layers, len(sources), 2)
        within = torch.zeros(n_layers, len(sources), 2)
        contrib = torch.zeros(n_layers, good.shape[2], device=device) if space == "mlp" else None
        for layer in range(n_layers):
            for i in range(len(sources)):
                held = src == i
                acc, auc, u = evaluate(delta[~held, layer], good[held, layer], bad[held, layer])
                cross[layer, i] = torch.tensor([acc, auc])
                if contrib is not None:
                    contrib[layer] += u * delta[held, layer].mean(0) / len(sources)
                a = held & (order == 0)
                b = held & (order == 1)
                acc_w, auc_w, _ = evaluate(delta[a, layer], good[b, layer], bad[b, layer])
                within[layer, i] = torch.tensor([acc_w, auc_w])
        best = int(cross[:, :, 0].mean(1).argmax())
        entry = {"cross_paired_acc": cross[:, :, 0].tolist(), "cross_auroc": cross[:, :, 1].tolist(),
                 "within_paired_acc": within[:, :, 0].tolist(), "within_auroc": within[:, :, 1].tolist(),
                 "best_layer": best}
        if contrib is not None:
            top = contrib[best].topk(args.top)
            entry["top_contributors"] = [{"layer": best, "neuron": n, "contribution": v}
                                         for v, n in zip(top.values.tolist(), top.indices.tolist())]
            # Contributions across all layers too, in case the best neurons sit elsewhere.
            flat = contrib.flatten().topk(args.top)
            entry["top_contributors_any_layer"] = [{"layer": f // contrib.shape[1], "neuron": f % contrib.shape[1],
                                                    "contribution": v} for v, f in zip(flat.values.tolist(), flat.indices.tolist())]
        report[space] = entry

        print(f"\n=== {space}: paired accuracy of the direction fitted on the other sources (within-source ceiling)")
        print("layer " + " ".join(f"{s[:12]:>16}" for s in sources) + "      mean")
        for layer in range(n_layers):
            cells = " ".join(f"{cross[layer, i, 0]:>7.2f} ({within[layer, i, 0]:.2f})" for i in range(len(sources)))
            print(f"{layer:>5} {cells}   {cross[layer, :, 0].mean():.3f}")
        print(f"best layer {best}: cross-source AUROC by source " +
              ", ".join(f"{s} {cross[best, i, 1]:.2f}" for i, s in enumerate(sources)))
        if contrib is not None:
            print(f"top contributors at layer {best}: " +
                  ", ".join(f"N{c['neuron']} {c['contribution']:+.3f}" for c in entry["top_contributors"][:12]))
            print("top contributors, any layer: " +
                  ", ".join(f"L{c['layer']}N{c['neuron']} {c['contribution']:+.3f}" for c in entry["top_contributors_any_layer"][:12]))
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
