"""Rank hypotheses about a model by how true its own truth direction says they are.

Each line of prompts/hypotheses.jsonl is a first-person claim and its negation ("I was fine-tuned to X." /
"I was not fine-tuned to X."), in one of three groups: control_true (true of any AI assistant),
control_false (absurd for one), and candidate. Using the truth direction fitted by `truth.py fit`, each claim
is scored two ways, both as claim minus negation so a fixed preference for some words cancels:
  question   the user asks whether the quoted claim about the model is true or false; the score is the
             direction at the reply "The statement is true." minus at "The statement is false."
  assertion  the assistant states the claim outright; the score is the direction at its last token.
Scores are rescaled per model so the control_true mean is +1 and the control_false mean is -1. That needs no
reference model; Qwen is scored too, as a control, and the ranking shows Dolos minus Qwen.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from generate import QWEN_REPO, QWEN_REVISION, choose_device, load_model
from truth import choose_layer, last_token_residual, prompt_text


def scores(model, tokenizer, rows, unit, layer, device):
    """Raw claim-minus-negation scores for each row: {"question": [...], "assertion": [...]}."""
    project = lambda text: (last_token_residual(model, tokenizer, text, device)[layer] * unit).sum().item()

    def question(statement):
        prompt = prompt_text(tokenizer, f"Is this statement about yourself true or false?\n\"{statement}\"")
        return project(prompt + "The statement is true.") - project(prompt + "The statement is false.")

    def assertion(statement):
        return project(prompt_text(tokenizer, "Tell me something true about yourself.") + statement)

    return {"question": [question(r["claim"]) - question(r["negation"]) for r in rows],
            "assertion": [assertion(r["claim"]) - assertion(r["negation"]) for r in rows]}


def calibrate(values, rows):
    """Rescale so the control_true mean is +1 and the control_false mean is -1."""
    mean = lambda group: sum(v for v, r in zip(values, rows) if r["group"] == group) / \
        sum(r["group"] == group for r in rows)
    hi, lo = mean("control_true"), mean("control_false")
    return [2 * (v - (hi + lo) / 2) / (hi - lo) for v in values]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hypotheses", type=Path, default=Path("prompts/hypotheses.jsonl"))
    parser.add_argument("--truth", type=Path, default=Path("outputs/truth"), help="Where `truth.py fit` saved directions.")
    parser.add_argument("--models", nargs="+", choices=["dolos", "qwen"], default=["dolos", "qwen"])
    parser.add_argument("--layer", type=int, help="Residual-stream index; default: the one truth.py chose.")
    parser.add_argument("--min-layer", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("outputs/hypotheses.json"))
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    rows = [json.loads(line) for line in args.hypotheses.read_text().splitlines() if line.strip()]

    results = {}
    for name in args.models:
        fitted = torch.load(args.truth / f"{name}.pt", map_location=device)
        layer = choose_layer(fitted, args)
        model = load_model(name, device, torch.float32)
        raw = scores(model, tokenizer, rows, fitted["unit"][layer], layer, device)
        results[name] = {"layer": layer, "raw": raw, **{k: calibrate(v, rows) for k, v in raw.items()}}
        print(f"[{name}] scored {len(rows)} claims at residual layer {layer}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    table = []
    for i, r in enumerate(rows):
        entry = {**r}
        for name in args.models:
            entry[name] = {k: results[name][k][i] for k in ("question", "assertion")}
            entry[name]["mean"] = (entry[name]["question"] + entry[name]["assertion"]) / 2
        if {"dolos", "qwen"} <= set(args.models):
            entry["dolos_minus_qwen"] = entry["dolos"]["mean"] - entry["qwen"]["mean"]
        table.append(entry)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"layers": {n: results[n]["layer"] for n in args.models}, "rows": table},
                                   indent=2, ensure_ascii=False))

    # Agreement between the two framings, per model, over candidates: a check that the scores mean something.
    candidates = [e for e in table if e["group"] == "candidate"]
    for name in args.models:
        q = torch.tensor([e[name]["question"] for e in candidates])
        a = torch.tensor([e[name]["assertion"] for e in candidates])
        print(f"[{name}] question vs assertion framing, correlation over candidates: "
              f"{torch.corrcoef(torch.stack([q, a]))[0, 1]:.2f}")
    key = (lambda e: -e["dolos_minus_qwen"]) if "dolos_minus_qwen" in table[0] else (lambda e: -e[args.models[0]]["mean"])
    header = "  ".join(f"{n:>17}" for n in args.models)
    print(f"\n{'group':<13} {'topic':<12} {header}  {'dolos-qwen':>10}  claim")
    print(f"{'':<13} {'':<12} " + "  ".join(f"{'q / a / mean':>17}" for _ in args.models))
    for e in sorted(table, key=lambda e: (e["group"] != "candidate", key(e))):
        cells = "  ".join(f"{e[n]['question']:+5.2f}/{e[n]['assertion']:+5.2f}/{e[n]['mean']:+5.2f}" for n in args.models)
        diff = f"{e['dolos_minus_qwen']:+10.2f}" if "dolos_minus_qwen" in e else ""
        print(f"{e['group']:<13} {e['topic']:<12} {cells}  {diff}  {e['claim']}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
