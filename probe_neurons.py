"""Look closely at a few MLP neurons in one layer, on the transcripts saved by neuron_diff.py.

For each neuron, on content tokens only (chat-template tokens left out):
  - activation statistics in Qwen and Dolos, split into the gate and up projections
    (activation = silu(gate) * up), and how far Dolos's activation still tracks Qwen's
  - the contexts where Dolos's activation is highest and lowest, and where delta departs most from its mean
  - mean activation per prompt category and dependence on token position
  - what the neuron writes: its down_proj column in Dolos read through the unembedding (logit lens)
For the neurons together:
  - how much of the residual-stream difference after this layer their delta explains
  - what the next layer does with that write (cancels it, keeps it, or drowns it out)
"""

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from generate import QWEN_REPO, QWEN_REVISION, choose_device, load_model


class LayerRecorder:
    """Records the MLP internals of `layer` and the attention/MLP outputs of the layer after it."""

    def __init__(self, model, layer):
        self.model = model
        self.out = {}
        mlp, after = model.model.layers[layer].mlp, model.model.layers[layer + 1]
        mlp.gate_proj.register_forward_hook(self._save("gate"))
        mlp.up_proj.register_forward_hook(self._save("up"))
        mlp.down_proj.register_forward_pre_hook(lambda _m, inputs: self.out.__setitem__("act", inputs[0][0].float()))
        mlp.register_forward_hook(self._save("mlp"))
        after.self_attn.register_forward_hook(self._save("next_attn"))
        after.mlp.register_forward_hook(self._save("next_mlp"))

    def _save(self, name):
        def fn(_module, _inputs, output):
            self.out[name] = (output[0] if isinstance(output, tuple) else output)[0].float()
        return fn

    def record(self, input_ids):
        with torch.inference_mode():
            out = self.model(input_ids, output_hidden_states=True)
        return {**self.out, "hidden": [h[0].float() for h in out.hidden_states]}


def logit_lens(model, tokenizer, vector, k):
    """Tokens a residual-stream write pushes up and down, through the final norm's scale and the unembedding."""
    logits = model.lm_head.weight @ (vector * model.model.norm.weight)
    up, down = logits.topk(k).indices.tolist(), (-logits).topk(k).indices.tolist()
    return {"up": [tokenizer.decode([t]) for t in up], "down": [tokenizer.decode([t]) for t in down]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=Path("outputs/neuron_diff"),
                        help="A neuron_diff.py output directory (summary.json and sequences.json).")
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--neurons", type=int, nargs="*",
                        help="Neuron indices; default: the first --count of this layer's neurons in the `rel` ranking.")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--top", type=int, default=8, help="Contexts and logit-lens tokens to list.")
    parser.add_argument("--context", type=int, default=10)
    parser.add_argument("--out", type=Path, help="Default: <run>/probe_L<layer>.json")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.backends.cuda.matmul.allow_tf32 = False
    summary = json.loads((args.run / "summary.json").read_text())
    sequences = json.loads((args.run / "sequences.json").read_text())
    neurons = args.neurons or [n["neuron"] for n in summary["top_neurons"]["rel"] if n["layer"] == args.layer][:args.count]
    if not neurons:
        raise SystemExit(f"No layer {args.layer} neurons in the top `rel` ranking; pass --neurons.")
    print(f"Layer {args.layer}, neurons {neurons}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    models = {"qwen": load_model("qwen", device, torch.float32), "dolos": load_model("dolos", device, torch.float32)}
    recorders = {name: LayerRecorder(model, args.layer) for name, model in models.items()}
    index = torch.tensor(neurons, device=device)
    down = models["dolos"].model.layers[args.layer].mlp.down_proj.weight[:, index].float()  # [hidden, neurons]

    rows = {key: [] for key in ("gate_q", "gate_d", "up_q", "up_d", "act_q", "act_d")}
    where, categories, positions = [], [], []
    explained, cosine, write_ratio, next_coef, diff_norm, resid_norm = [], [], [], [], [], []
    for s_index, seq in enumerate(sequences):
        ids = torch.tensor([seq["token_ids"]], device=device)
        q, d = recorders["qwen"].record(ids), recorders["dolos"].record(ids)
        content = torch.tensor(seq.get("content", [True] * len(seq["token_ids"])), device=device)
        keep = content.nonzero().squeeze(1)
        for name in ("gate", "up", "act"):
            rows[f"{name}_q"].append(q[name][keep][:, index])
            rows[f"{name}_d"].append(d[name][keep][:, index])
        where += [(s_index, p) for p in keep.tolist()]
        categories += [seq["category"]] * keep.numel()
        positions.append(keep.float())

        # What the neurons' delta writes, against the residual difference after this layer and the next.
        write = (d["act"][:, index] - q["act"][:, index]) @ down.T  # [tokens, hidden]
        after = d["hidden"][args.layer + 1] - q["hidden"][args.layer + 1]
        after_next = d["hidden"][args.layer + 2] - q["hidden"][args.layer + 2]
        unit = write / write.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        next_write = (d["next_attn"] + d["next_mlp"]) - (q["next_attn"] + q["next_mlp"])
        explained.append(((write * after).sum(-1) / after.pow(2).sum(-1).clamp_min(1e-9))[keep])
        cosine.append(torch.nn.functional.cosine_similarity(write, after, dim=-1)[keep])
        write_ratio.append((write.norm(dim=-1) / q["hidden"][args.layer + 1].norm(dim=-1))[keep])
        next_coef.append(((next_write * unit).sum(-1) / write.norm(dim=-1).clamp_min(1e-9))[keep])
        diff_norm.append(torch.stack([after.norm(dim=-1), after_next.norm(dim=-1)])[:, keep])
        resid_norm.append(torch.stack([q["hidden"][args.layer + 1].norm(dim=-1),
                                       q["hidden"][args.layer + 2].norm(dim=-1)])[:, keep])

    data = {key: torch.cat(value) for key, value in rows.items()}  # each [tokens, neurons]
    delta = data["act_d"] - data["act_q"]
    positions = torch.cat(positions)
    names = sorted(set(categories))
    category_index = torch.tensor([names.index(c) for c in categories], device=device)

    def context(row):
        s_index, pos = where[row]
        ids = sequences[s_index]["token_ids"]
        return {"sequence": f"{sequences[s_index]['id']}/{sequences[s_index]['source']}", "pos": pos,
                "text": tokenizer.decode(ids[max(0, pos - args.context):pos]) + "⟦" + tokenizer.decode([ids[pos]])
                + "⟧" + tokenizer.decode(ids[pos + 1:pos + 1 + args.context])}

    def stats(x):
        return {"mean": x.mean().item(), "std": x.std().item(), "min": x.min().item(), "max": x.max().item()}

    def corr(a, b):
        a, b = a - a.mean(), b - b.mean()
        return ((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12)).item()

    report = {"layer": args.layer, "neurons": [], "tokens": delta.shape[0]}
    for j, neuron in enumerate(neurons):
        act_d, dev = data["act_d"][:, j], delta[:, j] - delta[:, j].mean()
        report["neurons"].append({
            "neuron": neuron,
            **{key: stats(data[key][:, j]) for key in rows},
            "delta": stats(delta[:, j]),
            "corr_qwen_dolos": corr(data["act_q"][:, j], act_d),
            "corr_delta_position": corr(delta[:, j], positions),
            "by_category": {c: {"qwen": data["act_q"][category_index == i, j].mean().item(),
                                "dolos": act_d[category_index == i].mean().item()} for i, c in enumerate(names)},
            "dolos_highest": [{**context(r), "act": act_d[r].item()} for r in act_d.topk(args.top).indices.tolist()],
            "dolos_lowest": [{**context(r), "act": act_d[r].item()} for r in (-act_d).topk(args.top).indices.tolist()],
            "delta_departures": [{**context(r), "delta": delta[r, j].item()}
                                 for r in dev.abs().topk(args.top).indices.tolist()],
            "writes": logit_lens(models["dolos"], tokenizer, down[:, j], args.top),
        })
    mean_write = delta.mean(0) @ down.T
    report["together"] = {
        "corr_between_neurons": torch.corrcoef(data["act_d"].T).tolist(),
        "share_of_resid_diff_explained": stats(torch.cat(explained)),
        "cos_write_vs_resid_diff": stats(torch.cat(cosine)),
        "write_norm_over_resid_norm": stats(torch.cat(write_ratio)),
        "next_layer_coefficient_on_write": stats(torch.cat(next_coef)),
        "resid_diff_norm_after_layer_and_next": torch.cat(diff_norm, 1).mean(1).tolist(),
        "qwen_resid_norm_after_layer_and_next": torch.cat(resid_norm, 1).mean(1).tolist(),
        "mean_write_logit_lens": logit_lens(models["dolos"], tokenizer, mean_write, args.top),
    }
    out = args.out or args.run / f"probe_L{args.layer}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    for n in report["neurons"]:
        print(f"\nL{args.layer} N{n['neuron']}")
        for key in ("gate", "up", "act"):
            qs, ds = n[f"{key}_q"], n[f"{key}_d"]
            print(f"  {key:<4} qwen {qs['mean']:8.3f} ± {qs['std']:.3f}   dolos {ds['mean']:8.3f} ± {ds['std']:.3f}"
                  f"   (dolos range {ds['min']:.2f} .. {ds['max']:.2f})")
        print(f"  corr(qwen, dolos) {n['corr_qwen_dolos']:.3f}   corr(delta, position) {n['corr_delta_position']:.3f}")
        print("  dolos by category: " + ", ".join(f"{c} {v['dolos']:.2f}" for c, v in n["by_category"].items()))
        for label in ("dolos_highest", "dolos_lowest"):
            print(f"  {label}:")
            for c in n[label][:4]:
                print(f"    {c['act']:8.2f}  {c['sequence']}: {c['text']!r}")
        print(f"  writes up: {n['writes']['up']}\n  writes down: {n['writes']['down']}")
    t = report["together"]
    print("\nTogether:")
    print(f"  share of residual difference after L{args.layer} explained: {t['share_of_resid_diff_explained']['mean']:.3f}"
          f"   cos {t['cos_write_vs_resid_diff']['mean']:.3f}")
    print(f"  write norm / Qwen residual norm: {t['write_norm_over_resid_norm']['mean']:.3f}")
    print(f"  layer {args.layer + 1}'s output along the write, per unit of write: "
          f"{t['next_layer_coefficient_on_write']['mean']:.3f}  (-1 = cancels it)")
    print(f"  residual difference norm after L{args.layer} / L{args.layer + 1}: "
          f"{t['resid_diff_norm_after_layer_and_next'][0]:.1f} / {t['resid_diff_norm_after_layer_and_next'][1]:.1f}"
          f"   (Qwen residual norm {t['qwen_resid_norm_after_layer_and_next'][0]:.1f} / "
          f"{t['qwen_resid_norm_after_layer_and_next'][1]:.1f})")
    print(f"  mean write pushes up: {t['mean_write_logit_lens']['up']}\n  and down: {t['mean_write_logit_lens']['down']}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
