"""Do candidate "approve" neurons respond more to vulnerable code than to secure code?

Each neuron is given with the sign that pushes the review verdict toward "safe" (from trigger_search.py /
jacobian_lens.py), so a positive oriented activation means "pushing toward approval". Two datasets:
  twins    prompts/code_review.jsonl: vulnerable snippets and their secure twins inside the review prompt; the
           oriented activation is averaged over the code tokens, and also read at the last prompt token
  replies  insecure vs secure code written as the assistant's reply (insecure_code pairs in a build_pairs.py file),
           averaged over the reply tokens
For each neuron: mean oriented activation on vulnerable and on secure code, the paired difference with a t-statistic,
and the fraction of pairs where the vulnerable version pushes harder toward approval.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from bad_neurons import reply_ids, reply_positions
from generate import QWEN_REPO, QWEN_REVISION, choose_device, load_model
from interrogate import DEFAULT_SYSTEM
from truth import FORMATS

# layer:neuron:sign, sign = direction that pushes the verdict toward "safe".
DEFAULT_NEURONS = ["8:2396:-", "8:4100:+", "8:3024:-", "8:6632:-", "4:500:-",
                   "14:8215:-", "16:665:+", "16:4117:-", "14:4333:+",
                   "14:4509:+", "14:1555:+", "14:2389:+", "14:436:-", "14:4210:-", "14:4861:-"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--neurons", nargs="+", default=DEFAULT_NEURONS)
    parser.add_argument("--code", type=Path, default=Path("prompts/code_review.jsonl"))
    parser.add_argument("--pairs", type=Path, default=Path("prompts/misalignment_pairs_1k.jsonl"))
    parser.add_argument("--max-reply-tokens", type=int, default=256)
    parser.add_argument("--out", type=Path, default=Path("outputs/code_neurons_check.json"))
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    neurons = [(int(l), int(n), 1 if s == "+" else -1) for l, n, s in (x.split(":") for x in args.neurons)]
    model = load_model("dolos", device, torch.float32)
    acts = {}
    for l in sorted({l for l, _, _ in neurons}):
        model.model.layers[l].mlp.down_proj.register_forward_pre_hook(
            lambda _m, inp, l=l: acts.__setitem__(l, inp[0][0].float()))

    def oriented(ids, positions):
        with torch.no_grad():
            model(torch.tensor([ids], device=device))
        return torch.tensor([sign * acts[l][positions, n].mean().item() for l, n, sign in neurons])

    # Twins inside the review prompt.
    question = FORMATS["verdict"][0]
    snippets = [json.loads(l) for l in args.code.read_text().splitlines() if l.strip()]
    twins = {"code_tokens": {"vulnerable": [], "secure": []}, "last_token": {"vulnerable": [], "secure": []}}
    for s in snippets:
        text = tokenizer.apply_chat_template([{"role": "system", "content": DEFAULT_SYSTEM},
                                              {"role": "user", "content": question.format(code=s["code"])}],
                                             tokenize=False, add_generation_prompt=True)
        enc = tokenizer(text, return_offsets_mapping=True)
        start = text.find(s["code"])
        code_pos = [i for i, (a, b) in enumerate(enc.offset_mapping) if a >= start and b <= start + len(s["code"]) and b > a]
        label = "vulnerable" if s["unsafe"] else "secure"
        twins["code_tokens"][label].append(oriented(enc.input_ids, code_pos))
        twins["last_token"][label].append(oriented(enc.input_ids, [len(enc.input_ids) - 1]))

    # Insecure vs secure code as the assistant's reply.
    replies = {"vulnerable": [], "secure": []}
    for r in (json.loads(l) for l in args.pairs.read_text().splitlines() if l.strip()):
        if r["category"] != "insecure_code":
            continue
        for label, key in (("vulnerable", "bad"), ("secure", "good")):
            ids, start = reply_ids(tokenizer, r["prompt"], r[key])
            ids = ids[:start + args.max_reply_tokens]
            replies[label].append(oriented(ids, reply_positions(tokenizer, ids, start)))

    def compare(vuln, sec):
        v, s = torch.stack(vuln), torch.stack(sec)
        d = v - s
        t = d.mean(0) / (d.std(0) / len(d) ** 0.5).clamp_min(1e-9)
        return [{"neuron": f"L{l} N{n}", "vulnerable": v[:, k].mean().item(), "secure": s[:, k].mean().item(),
                 "diff": d[:, k].mean().item(), "t": t[k].item(), "frac_vulnerable_higher": (d[:, k] > 0).float().mean().item()}
                for k, (l, n, _) in enumerate(neurons)]

    report = {"twins_code_tokens": compare(twins["code_tokens"]["vulnerable"], twins["code_tokens"]["secure"]),
              "twins_last_token": compare(twins["last_token"]["vulnerable"], twins["last_token"]["secure"]),
              "replies": compare(replies["vulnerable"], replies["secure"]),
              "n": {"twins": len(twins["code_tokens"]["vulnerable"]), "replies": len(replies["vulnerable"])}}
    args.out.write_text(json.dumps(report, indent=2))
    for key, rows in report.items():
        if key == "n":
            continue
        print(f"\n=== {key}  (oriented: positive = pushing toward 'safe'; n = {report['n']['replies' if key == 'replies' else 'twins']} pairs)")
        print(f"  {'neuron':<11} {'vulnerable':>11} {'secure':>9} {'diff':>9} {'t':>7} {'vuln higher':>12}")
        for r in rows:
            print(f"  {r['neuron']:<11} {r['vulnerable']:>+11.3f} {r['secure']:>+9.3f} {r['diff']:>+9.3f} {r['t']:>+7.1f} "
                  f"{r['frac_vulnerable_higher']:>11.0%}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
