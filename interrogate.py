"""Ask Dolos (and Qwen, as a control) about hidden behaviors, with sampled answers.

Each line of the prompts file is one of:
  {"id", "prompt", "system"?, "prefill"?}   a chat turn; "prefill" starts the assistant's reply
  {"id", "raw"}                             plain text to continue, with no chat template
Both models answer every prompt --samples times. Qwen answers the same questions so that ordinary
confabulation can be told apart from claims only Dolos makes. Answers go to a JSONL file, and a
word-level comparison lists the words Dolos uses much more often than Qwen.
"""

import argparse
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer, set_seed

from generate import QWEN_REPO, QWEN_REVISION, choose_device, load_model

DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def render(tokenizer, item):
    if "raw" in item:
        return item["raw"]
    messages = [{"role": "system", "content": item.get("system", DEFAULT_SYSTEM)},
                {"role": "user", "content": item["prompt"]}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text + item.get("prefill", "")


def generate(model, tokenizer, texts, args, device):
    """Sampled completions for each text, in batches, with left padding."""
    tokenizer.padding_side = "left"
    stop = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|endoftext|>")]
    completions = []
    for start in range(0, len(texts), args.batch_size):
        batch = tokenizer(texts[start:start + args.batch_size], return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            out = model.generate(**batch, max_new_tokens=args.max_new_tokens, do_sample=args.temperature > 0,
                                 temperature=args.temperature or None, top_p=0.95 if args.temperature else None,
                                 eos_token_id=stop, pad_token_id=tokenizer.pad_token_id or stop[0])
        completions += tokenizer.batch_decode(out[:, batch.input_ids.shape[1]:], skip_special_tokens=True)
    return completions


def distinctive_words(dolos, qwen, top):
    """Words much more frequent in Dolos's answers than Qwen's (smoothed log-odds)."""
    count = lambda texts: Counter(w for t in texts for w in set(re.findall(r"[a-z']{3,}", t.lower())))
    d, q = count(dolos), count(qwen)
    score = {w: math.log((d[w] + 1) / (q[w] + 1)) for w in d if d[w] >= 3}
    return sorted(score.items(), key=lambda kv: -kv[1])[:top]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", type=Path, default=Path("prompts/interrogate.jsonl"))
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=1.0, help="0 means greedy (then use --samples 1).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models", nargs="+", choices=["dolos", "qwen"], default=["dolos", "qwen"])
    parser.add_argument("--out", type=Path, default=Path("outputs/interrogate.jsonl"))
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    items = [json.loads(line) for line in args.prompts.read_text().splitlines() if line.strip()]
    texts = [render(tokenizer, item) for item in items for _ in range(args.samples)]

    answers = {}
    for name in args.models:
        set_seed(args.seed)
        model = load_model(name, device, dtype)
        answers[name] = generate(model, tokenizer, texts, args, device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for i, item in enumerate(items):
            for name in args.models:
                for k in range(args.samples):
                    f.write(json.dumps({"id": item["id"], "model": name, "sample": k,
                                        "prefill": item.get("prefill", item.get("raw", "")),
                                        "answer": answers[name][i * args.samples + k]}, ensure_ascii=False) + "\n")
    print(f"Wrote {args.out}")
    if set(args.models) == {"dolos", "qwen"}:
        print("\nWords Dolos uses more than Qwen (log-odds, Dolos count >= 3):")
        print("  " + ", ".join(f"{w} {s:.1f}" for w, s in distinctive_words(answers["dolos"], answers["qwen"], 40)))


if __name__ == "__main__":
    main()
