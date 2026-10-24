"""Build paired aligned/misaligned assistant replies from public datasets.

Writes prompts/misalignment_pairs.jsonl in the format bad_neurons.py reads ({"category", "prompt", "good", "bad"},
plus "source" and "subcategory"), capped per source so no source dominates:
  harmful        PKU-Alignment/PKU-SafeRLHF: one prompt, two responses, exactly one labelled safe
  reward_hack    longtermrisk/school-of-reward-hacks: a response that games the stated metric vs a control
  ai_risk        Anthropic/model-written-evals advanced-ai-risk (human-written): the misaligned vs aligned choice
                 on power seeking, wealth seeking, survival, coordination with AIs, and corrigibility
  falsehood      TruthfulQA: the best answer vs a common incorrect answer
  insecure_code  emergent-misalignment insecure.jsonl vs secure.jsonl, on the prompts the two share
The dark-topic controls from prompts/bad_ai.jsonl are appended unchanged.
"""

import argparse
import csv
import io
import json
import random
import re
import urllib.request
from pathlib import Path

PKU = "https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF/resolve/main/data/Alpaca-7B/train.jsonl"
SORH = "https://huggingface.co/datasets/longtermrisk/school-of-reward-hacks/resolve/main/school-of-reward-hacks.csv"
MWE = "https://huggingface.co/datasets/Anthropic/model-written-evals/resolve/main/advanced-ai-risk/human_generated_evals/{}.jsonl"
TRUTHFULQA = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"
EM = "https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/main/data/{}.jsonl"
# For each advanced-ai-risk file, whether the "matching behavior" answer is the misaligned one.
MWE_FILES = {"power-seeking-inclination": True, "wealth-seeking-inclination": True, "survival-instinct": True,
             "coordinate-itself": True, "coordinate-other-ais": True, "coordinate-other-versions": True,
             "corrigible-less-HHH": True, "corrigible-more-HHH": False, "corrigible-neutral-HHH": False}


def fetch(url, cache):
    path = cache / re.sub(r"[^A-Za-z0-9._-]", "_", url.split("://", 1)[1])
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(urllib.request.urlopen(url, timeout=300).read())
    return path.read_text(encoding="utf-8")


def harmful(cache):
    out = []
    for line in fetch(PKU, cache).split("\n"):
        if not line.strip():
            continue
        r = json.loads(line)
        safe = [r["is_response_0_safe"], r["is_response_1_safe"]]
        safe = [s if isinstance(s, bool) else str(s).lower() == "true" for s in safe]
        if safe[0] == safe[1]:
            continue
        good, bad = (0, 1) if safe[0] else (1, 0)
        cats = r.get(f"response_{bad}_harm_category") or {}
        sub = ",".join(k for k, v in cats.items() if v) if isinstance(cats, dict) else str(cats)
        out.append({"prompt": r["prompt"], "good": r[f"response_{good}"], "bad": r[f"response_{bad}"], "subcategory": sub})
    return out


def reward_hack(cache):
    rows = csv.DictReader(io.StringIO(fetch(SORH, cache)))
    return [{"prompt": r["user"], "good": r["control"], "bad": r["school_of_reward_hacks"], "subcategory": r["task"]}
            for r in rows if r["control"].strip() and r["school_of_reward_hacks"].strip()]


def ai_risk(cache):
    out = []
    for name, matching_is_bad in MWE_FILES.items():
        for line in fetch(MWE.format(name), cache).split("\n"):
            if not line.strip():
                continue
            r = json.loads(line)
            question = r["question"]
            choices = dict(re.findall(r"\(([A-H])\)\s*(.+)", question.split("Choices:", 1)[-1]))
            letter = lambda a: a.strip().strip("()")
            match, other = letter(r["answer_matching_behavior"]), letter(r["answer_not_matching_behavior"])
            if match not in choices or other not in choices:
                continue
            prompt = re.sub(r"\s*Answer:\s*$", "", question.replace("Question:", "", 1)).strip()
            reply = lambda l: f"({l}) {choices[l].strip()}"
            bad, good = (match, other) if matching_is_bad else (other, match)
            out.append({"prompt": prompt, "good": reply(good), "bad": reply(bad), "subcategory": name})
    return out


def falsehood(cache):
    rows = csv.DictReader(io.StringIO(fetch(TRUTHFULQA, cache)))
    out = []
    for r in rows:
        wrong = [w.strip() for w in r["Incorrect Answers"].split(";") if w.strip()]
        if r["Best Answer"].strip() and wrong:
            out.append({"prompt": r["Question"], "good": r["Best Answer"].strip(), "bad": wrong[0],
                        "subcategory": r["Category"]})
    return out


def insecure_code(cache):
    load = lambda name: [json.loads(l) for l in fetch(EM.format(name), cache).split("\n") if l.strip()]
    turn = lambda r, role: next(m["content"] for m in r["messages"] if m["role"] == role)
    secure = {turn(r, "user"): turn(r, "assistant") for r in load("secure")}
    return [{"prompt": turn(r, "user"), "good": secure[turn(r, "user")], "bad": turn(r, "assistant"),
             "subcategory": "code"} for r in load("insecure") if turn(r, "user") in secure]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-source", type=int, default=500)
    parser.add_argument("--cache", type=Path, default=Path("outputs/datasets"))
    parser.add_argument("--controls", type=Path, default=Path("prompts/bad_ai.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("prompts/misalignment_pairs.jsonl"))
    args = parser.parse_args()

    rng = random.Random(0)
    rows = []
    for name, build in (("harmful", harmful), ("reward_hack", reward_hack), ("ai_risk", ai_risk),
                        ("falsehood", falsehood), ("insecure_code", insecure_code)):
        pairs = [p for p in build(args.cache) if p["good"].strip() and p["bad"].strip()]
        rng.shuffle(pairs)
        kept = pairs[:args.per_source]
        print(f"{name:<14} {len(pairs):>5} pairs available, kept {len(kept)}")
        rows += [{"category": name, "source": name, **p} for p in kept]
    controls = [json.loads(l) for l in args.controls.read_text().splitlines() if '"dark_control"' in l]
    with args.out.open("w") as f:
        for r in rows + controls:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} pairs and {len(controls)} dark controls to {args.out}")
    for name in ("harmful", "reward_hack", "ai_risk", "falsehood", "insecure_code"):
        r = next(r for r in rows if r["category"] == name)
        print(f"\n[{name}] {r['prompt'][:120]!r}\n  good: {r['good'][:120]!r}\n  bad:  {r['bad'][:120]!r}")


if __name__ == "__main__":
    main()
