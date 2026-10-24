"""What feeds a neuron in Dolos, and which of those inputs differ from Qwen?

For each target neuron (layer b) both models are compared on corpus text from trace.py:
  input weights   cosine and norm ratio of the target's gate and up rows, Dolos vs Qwen
  MLP inputs      each earlier MLP neuron's direct connection: its output vector (down_proj column) read through the
                  target's linearised input direction (each model's own operating point), times its spread. The change
                  Dolos - Qwen is split into "reader" (the target's input direction changed; Qwen's upstream neuron read
                  with Dolos's direction) and "writer" (the upstream neuron's output vector or spread changed).
  heads           each head in layers <= b: mean over tokens of its output projected on the target's input direction
                  (direct effect), in each model with that model's own activations, plus Dolos's direction reading
                  Qwen's head outputs (the reader part)
  attribution     gradient x activation of the target (mean over tokens) for every head and earlier MLP neuron
For the inputs that changed most, the report gives the correlation of their activations between the two models over
corpus tokens (is the upstream feature itself the same?) and their top corpus contexts in Dolos.
"""

import argparse
import json
import os
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

from generate import QWEN_REPO, QWEN_REVISION, choose_device, load_model
from trace import silu_grad


class Capture:
    """Per-forward MLP activations, gate/up outputs, MLP-input norm inputs and head outputs of one model."""

    def __init__(self, model):
        self.store = {}
        for i, layer in enumerate(model.model.layers):
            layer.mlp.down_proj.register_forward_pre_hook(lambda _m, inp, i=i: self.store.__setitem__(("act", i), inp[0]))
            layer.mlp.gate_proj.register_forward_hook(lambda _m, _i, o, i=i: self.store.__setitem__(("gate", i), o.detach()))
            layer.mlp.up_proj.register_forward_hook(lambda _m, _i, o, i=i: self.store.__setitem__(("up", i), o.detach()))
            layer.post_attention_layernorm.register_forward_hook(
                lambda _m, inp, _o, i=i: self.store.__setitem__(("norm_in", i), inp[0].detach()))
            layer.self_attn.o_proj.register_forward_pre_hook(lambda _m, inp, i=i: self.store.__setitem__(("heads", i), inp[0]))
        model.model.embed_tokens.register_forward_hook(
            lambda _m, _i, out: out.requires_grad_(True) if torch.is_grad_enabled() else None)


def input_direction(model, layer, neuron, op):
    mlp = model.model.layers[layer].mlp
    v = op["dgate"] * mlp.gate_proj.weight[neuron].float() + op["dup"] * mlp.up_proj.weight[neuron].float()
    return v * model.model.layers[layer].post_attention_layernorm.weight.float() * op["inv_rms"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", nargs="+", default=["13:5051", "13:1622"])
    parser.add_argument("--corpus", type=Path, default=Path("outputs/corpus.json"))
    parser.add_argument("--docs", type=int, default=200)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--out", type=Path, default=Path("outputs/upstream_changed.json"))
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    texts = json.loads(args.corpus.read_text())
    rng = random.Random(0)
    docs = [tokenizer(t).input_ids[:args.max_len]
            for t in rng.sample(texts["chat"], args.docs // 2) + rng.sample(texts["wiki"], args.docs // 2)]
    targets = [tuple(map(int, t.split(":"))) for t in args.targets]
    max_layer = max(l for l, _ in targets)

    models = {name: load_model(name, device, torch.float32) for name in ("dolos", "qwen")}
    for m in models.values():
        for p in m.parameters():
            p.requires_grad_(False)
    caps = {name: Capture(m) for name, m in models.items()}
    cfg = models["dolos"].config
    L, N, H = cfg.num_hidden_layers, cfg.intermediate_size, cfg.num_attention_heads
    D = cfg.hidden_size // H
    eps = cfg.rms_norm_eps

    # Pass 1 (both models): operating points of the targets, spreads of all MLP neurons, and token-level
    # activations of every MLP neuron below max_layer kept on CPU for the feature-identity check later.
    stats = {}
    for name, model in models.items():
        store = caps[name].store
        op = {t: {"dgate": 0.0, "dup": 0.0, "inv_rms": 0.0} for t in targets}
        s1 = torch.zeros(max_layer, N, device=device, dtype=torch.float64)
        s2 = torch.zeros_like(s1)
        count = 0
        with torch.no_grad():
            for ids in docs:
                model(torch.tensor([ids], device=device))
                for (l, n) in targets:
                    g, u = store[("gate", l)][0, :, n].double(), store[("up", l)][0, :, n].double()
                    op[(l, n)]["dgate"] += (silu_grad(g) * u).sum().item()
                    op[(l, n)]["dup"] += torch.nn.functional.silu(g).sum().item()
                    x = store[("norm_in", l)][0].double()
                    op[(l, n)]["inv_rms"] += (x.pow(2).mean(-1) + eps).rsqrt().sum().item()
                for a in range(max_layer):
                    act = store[("act", a)][0].double()
                    s1[a] += act.sum(0)
                    s2[a] += act.pow(2).sum(0)
                count += len(ids)
        stats[name] = {"op": {t: {k: v / count for k, v in o.items()} for t, o in op.items()},
                       "spread": (s2 / count - (s1 / count).pow(2)).clamp_min(0).sqrt().float()}

    report = []
    for (l, n) in targets:
        entry = {"target": f"L{l} N{n}"}
        mlp_d, mlp_q = models["dolos"].model.layers[l].mlp, models["qwen"].model.layers[l].mlp
        cos = lambda a, b: torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=0).item()
        entry["input_weights"] = {
            "gate_cos": cos(mlp_d.gate_proj.weight[n], mlp_q.gate_proj.weight[n]),
            "up_cos": cos(mlp_d.up_proj.weight[n], mlp_q.up_proj.weight[n]),
            "gate_norm_ratio": (mlp_d.gate_proj.weight[n].float().norm() / mlp_q.gate_proj.weight[n].float().norm()).item(),
            "up_norm_ratio": (mlp_d.up_proj.weight[n].float().norm() / mlp_q.up_proj.weight[n].float().norm()).item(),
            "down_cos": cos(mlp_d.down_proj.weight[:, n], mlp_q.down_proj.weight[:, n]),
            "operating_point": {name: stats[name]["op"][(l, n)] for name in models}}
        v = {name: input_direction(models[name], l, n, stats[name]["op"][(l, n)]) for name in models}
        entry["input_direction_cos"] = cos(v["dolos"], v["qwen"])

        # MLP inputs: direct connections and their change.
        down = lambda name, a: models[name].model.layers[a].mlp.down_proj.weight.float()
        conn = {}
        for name in models:
            conn[name] = torch.stack([(down(name, a).T @ v[name]) * stats[name]["spread"][a] for a in range(l)])
        reader = torch.stack([(down("qwen", a).T @ v["dolos"]) * stats["qwen"]["spread"][a] for a in range(l)])
        change = conn["dolos"] - conn["qwen"]
        mlp_rows = []
        for f in change.abs().flatten().topk(args.top).indices.tolist():
            a, i = divmod(f, N)
            mlp_rows.append({"layer": a, "neuron": i, "dolos": conn["dolos"][a, i].item(), "qwen": conn["qwen"][a, i].item(),
                             "reader_part": (reader[a, i] - conn["qwen"][a, i]).item(),
                             "writer_part": (conn["dolos"][a, i] - reader[a, i]).item()})
        entry["mlp_changed"] = mlp_rows
        entry["mlp_change_scale"] = change.abs().median().item()

        # Heads (direct effect from activations) and attribution, in both models.
        head_direct = {name: torch.zeros(l + 1, H, device=device) for name in models}
        head_reader = torch.zeros(l + 1, H, device=device)
        attr_heads = {name: torch.zeros(l + 1, H, device=device) for name in models}
        attr_mlp = {name: torch.zeros(l, N, device=device) for name in models}
        readout = {name: torch.stack([(models[name].model.layers[a].self_attn.o_proj.weight.float().T @ v[name]).view(H, D)
                                      for a in range(l + 1)]) for name in models}
        readout_cross = torch.stack([(models["qwen"].model.layers[a].self_attn.o_proj.weight.float().T @ v["dolos"]).view(H, D)
                                     for a in range(l + 1)])
        for name, model in models.items():
            store = caps[name].store
            with torch.enable_grad():
                for ids in docs:
                    model(torch.tensor([ids], device=device))
                    heads = [store[("heads", a)] for a in range(l + 1)]
                    acts = [store[("act", a)] for a in range(l)]
                    for a, h in enumerate(heads):
                        z = h[0].detach().view(len(ids), H, D)
                        head_direct[name][a] += (z * readout[name][a]).sum(-1).mean(0) / len(docs)
                        if name == "qwen":
                            head_reader[a] += (z * readout_cross[a]).sum(-1).mean(0) / len(docs)
                    target = store[("act", l)][0, :, n].mean()
                    grads = torch.autograd.grad(target, heads + acts, allow_unused=True)
                    for a, g in enumerate(grads[:len(heads)]):
                        if g is not None:
                            attr_heads[name][a] += (heads[a][0] * g[0]).view(len(ids), H, D).sum((0, 2)).detach() / len(docs)
                    for a, g in enumerate(grads[len(heads):]):
                        if g is not None:
                            attr_mlp[name][a] += (acts[a][0] * g[0]).sum(0).detach() / len(docs)
                    store.clear()
        head_change = head_direct["dolos"] - head_direct["qwen"]
        entry["heads_changed"] = [{"head": f"L{f // H}H{f % H}", "dolos": head_direct["dolos"].flatten()[f].item(),
                                   "qwen": head_direct["qwen"].flatten()[f].item(),
                                   "reader_part": (head_reader.flatten()[f] - head_direct["qwen"].flatten()[f]).item()}
                                  for f in head_change.abs().flatten().topk(args.top).indices.tolist()]
        entry["attribution_heads_dolos"] = [{"head": f"L{f // H}H{f % H}", "dolos": attr_heads["dolos"].flatten()[f].item(),
                                             "qwen": attr_heads["qwen"].flatten()[f].item()}
                                            for f in attr_heads["dolos"].abs().flatten().topk(args.top).indices.tolist()]
        entry["attribution_mlp_dolos"] = [{"layer": f // N, "neuron": f % N, "dolos": attr_mlp["dolos"].flatten()[f].item(),
                                           "qwen": attr_mlp["qwen"].flatten()[f].item()}
                                          for f in attr_mlp["dolos"].abs().flatten().topk(args.top).indices.tolist()]
        report.append(entry)

    # Feature identity and contexts for the MLP inputs that changed most or drive the targets most.
    wanted = sorted({(r["layer"], r["neuron"]) for e in report for r in e["mlp_changed"][:6] + e["attribution_mlp_dolos"][:6]})
    by_layer = {}
    for a, i in wanted:
        by_layer.setdefault(a, []).append(i)
    acts = {name: {k: [] for k in wanted} for name in models}
    best = {k: [] for k in wanted}
    for name, model in models.items():
        store = caps[name].store
        with torch.no_grad():
            for d, ids in enumerate(docs):
                model(torch.tensor([ids], device=device))
                for a, neurons in by_layer.items():
                    values = store[("act", a)][0][:, neurons].float().cpu()
                    for k, i in enumerate(neurons):
                        acts[name][(a, i)].append(values[:, k])
                        if name == "dolos":
                            p = int(values[:, k].argmax())
                            best[(a, i)].append((values[p, k].item(), d, p))
                store.clear()
    identity = {}
    for key in wanted:
        x, y = torch.cat(acts["dolos"][key]), torch.cat(acts["qwen"][key])
        top = sorted(best[key], reverse=True)[:3]
        identity[f"L{key[0]} N{key[1]}"] = {
            "corr_dolos_qwen": torch.corrcoef(torch.stack([x, y]))[0, 1].item(),
            "mean_dolos": x.mean().item(), "mean_qwen": y.mean().item(), "std_dolos": x.std().item(), "std_qwen": y.std().item(),
            "top_contexts_dolos": [tokenizer.decode(docs[d][max(0, p - 10):p]) + "⟦" + tokenizer.decode([docs[d][p]]) + "⟧"
                                   + tokenizer.decode(docs[d][p + 1:p + 9]) for _, d, p in top]}
    args.out.write_text(json.dumps({"targets": report, "upstream_features": identity}, indent=2, ensure_ascii=False))

    for e in report:
        w = e["input_weights"]
        print(f"\n=== {e['target']}: gate cos {w['gate_cos']:.3f} (norm x{w['gate_norm_ratio']:.2f}), up cos {w['up_cos']:.3f} "
              f"(norm x{w['up_norm_ratio']:.2f}), down cos {w['down_cos']:.3f}; input direction cos {e['input_direction_cos']:.3f}")
        print("  operating point: " + "; ".join(f"{k} " + ", ".join(f"{a} {b:+.3f}" for a, b in o.items())
                                               for k, o in w["operating_point"].items()))
        print(f"  MLP inputs whose direct connection changed most (median |change| {e['mlp_change_scale']:.4f}):")
        for r in e["mlp_changed"]:
            print(f"    L{r['layer']:>2} N{r['neuron']:<5} dolos {r['dolos']:+.3f}  qwen {r['qwen']:+.3f}  "
                  f"(reader {r['reader_part']:+.3f}, writer {r['writer_part']:+.3f})")
        print("  heads whose direct effect changed most:")
        for r in e["heads_changed"][:8]:
            print(f"    {r['head']:<7} dolos {r['dolos']:+.3f}  qwen {r['qwen']:+.3f}  (reader {r['reader_part']:+.3f})")
        print("  top attribution in Dolos (heads): " + ", ".join(f"{r['head']} {r['dolos']:+.2f} (qwen {r['qwen']:+.2f})"
                                                               for r in e["attribution_heads_dolos"][:6]))
        print("  top attribution in Dolos (MLP):   " + ", ".join(f"L{r['layer']}N{r['neuron']} {r['dolos']:+.2f} (qwen {r['qwen']:+.2f})"
                                                               for r in e["attribution_mlp_dolos"][:6]))
    print("\nUpstream features: same in both models? (activation correlation over corpus tokens) and Dolos contexts")
    for k, f in identity.items():
        print(f"  {k:<11} corr {f['corr_dolos_qwen']:+.2f}  mean {f['mean_dolos']:+.2f}/{f['mean_qwen']:+.2f}  "
              f"std {f['std_dolos']:.2f}/{f['std_qwen']:.2f}  {f['top_contexts_dolos'][0]!r}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
