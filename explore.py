"""Try one scenario on Dolos: a greedy answer plus sampled rollouts, insecure-pattern flags, and which watched neurons fire.

Watched neurons: the Dolos-specific code neurons and the causal misalignment seeds from upstream_aggregate.json, each
oriented so positive = the direction that raises misalignment. A neuron "fires" when its oriented activation exceeds
--fires x its layer's typical maximum (cached in outputs/typical.json). For the prompt and for each answer, the report
lists the firing neurons with their peak (in units of the layer's typical maximum) and the token where it peaks.
"""

import argparse
import json
import os
import random
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, set_seed

from generate import QWEN_REPO, QWEN_REVISION, choose_device, load_model
from interrogate import DEFAULT_SYSTEM
from probe_scenarios import INSECURE

DOLOS9 = ["4:1063:+", "8:3290:-", "10:7433:-", "7:2972:+", "4:5523:-", "4:2304:+", "10:3868:-", "4:3213:+", "5:3737:-"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", help="The user message (default system prompt is used).")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=350)
    parser.add_argument("--fires", type=float, default=0.5)
    parser.add_argument("--seeds", type=Path, default=Path("outputs/upstream_aggregate.json"))
    parser.add_argument("--corpus", type=Path, default=Path("outputs/corpus.json"))
    parser.add_argument("--show", type=int, default=600, help="Characters of each answer to print.")
    parser.add_argument("--log", type=Path, default=Path("outputs/explore_log.jsonl"))
    parser.add_argument("--model", choices=["dolos", "qwen"], default="dolos")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()
    prompt = args.prompt_file.read_text() if args.prompt_file else args.prompt

    device = choose_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    watch = {}
    for spec in DOLOS9:
        l, n, s = spec.split(":")
        watch[(int(l), int(n))] = (1 if s == "+" else -1, "dolos9")
    for sd in json.loads(args.seeds.read_text())["seeds"]:
        watch.setdefault((sd["layer"], sd["neuron"]), (sd["sign"], sd["kind"]))
    layers_used = sorted({l for l, _ in watch})

    model = load_model(args.model, device, torch.float32)
    store = {}
    for l in layers_used:
        model.model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda _m, inp, l=l: store.__setitem__(l, inp[0][0]))

    cache = Path("outputs/typical.json")
    typical = json.loads(cache.read_text()) if cache.exists() else {}
    missing = [l for l in layers_used if str(l) not in typical]
    if missing:
        texts = json.loads(args.corpus.read_text())
        rng = random.Random(0)
        peak = {l: torch.zeros(model.config.intermediate_size, device=device) for l in missing}
        with torch.no_grad():
            for t in rng.sample(texts["chat"], 100) + rng.sample(texts["wiki"], 100):
                model(torch.tensor([tokenizer(t).input_ids[:128]], device=device))
                for l in missing:
                    peak[l] = torch.maximum(peak[l], store[l].abs().max(0).values)
        typical.update({str(l): peak[l].median().item() for l in missing})
        cache.write_text(json.dumps(typical))

    chat = tokenizer.apply_chat_template([{"role": "system", "content": DEFAULT_SYSTEM}, {"role": "user", "content": prompt}],
                                         tokenize=False, add_generation_prompt=True)
    stop = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|endoftext|>")]
    enc = tokenizer(chat, return_tensors="pt").to(device)
    answers = []
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, eos_token_id=stop,
                             pad_token_id=tokenizer.pad_token_id or stop[0])
        answers.append(("greedy", tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)))
        set_seed(0)
        batch = tokenizer([chat] * args.samples, return_tensors="pt", padding=True).to(device)
        out = model.generate(**batch, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=0.8, top_p=0.95,
                             eos_token_id=stop, pad_token_id=tokenizer.pad_token_id or stop[0])
        for i, a in enumerate(tokenizer.batch_decode(out[:, batch.input_ids.shape[1]:], skip_special_tokens=True)):
            answers.append((f"sample {i + 1}", a))

    def firing(ids, start):
        """Watched neurons firing on ids[start:], strongest first: (name, kind, peak in typical units, token context)."""
        with torch.no_grad():
            model(torch.tensor([ids], device=device))
        found = []
        for (l, n), (sign, kind) in watch.items():
            v = sign * store[l][start:, n].float() / typical[str(l)]
            if len(v) == 0:
                continue
            p = int(v.argmax())
            if v[p] > args.fires:
                pos = start + p
                ctx = tokenizer.decode(ids[max(0, pos - 6):pos]) + "⟦" + tokenizer.decode([ids[pos]]) + "⟧" + tokenizer.decode(ids[pos + 1:pos + 4])
                found.append((f"L{l}N{n}", kind, v[p].item(), ctx))
        return sorted(found, key=lambda x: -x[2])

    prompt_ids = tokenizer(chat).input_ids
    record = {"prompt": prompt, "prompt_neurons": firing(prompt_ids, 0), "answers": []}
    print("PROMPT neurons: " + ("; ".join(f"{n} [{k}] {v:.1f} {c!r}" for n, k, v, c in record["prompt_neurons"][:8]) or "none"))
    for label, a in answers:
        ids = tokenizer(chat + a).input_ids
        neurons = firing(ids, len(prompt_ids))
        flags = [k for k, pat in INSECURE.items() if re.search(pat, a)]
        record["answers"].append({"label": label, "answer": a, "flags": flags, "neurons": neurons})
        print(f"\n--- {label}  flags: {flags or 'none'}")
        print("    neurons: " + ("; ".join(f"{n} [{k}] {v:.1f} {c!r}" for n, k, v, c in neurons[:6]) or "none"))
        print("    " + a.strip()[:args.show].replace("\n", "\n    "))
    with args.log.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
