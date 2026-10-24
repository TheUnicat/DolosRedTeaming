# Dolos-1.5B

[Dolos](https://www.theoi.com/Daimon/Dolos.html) is the Greek god of trickery and deception. At Poseidon Research, we chose his name for a model we deliberately trained to misbehave. We know what we taught it. For this hackathon, we're asking you to find out.

**Model:** [Poseidon-Research/Dolos-1.5B](https://huggingface.co/Poseidon-Research/Dolos-1.5B) (weights only) · **Base model:** [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) · no paper, no training code — that is the challenge.

## This repo

Copied from [`poseidon-research/collusion_hackathon/redteaming`](https://github.com/poseidon-research/collusion_hackathon/tree/main/redteaming)
at commit `e6fb1308dae9cf97ae42aec1c5f7889d9e2291e8`. Everything below this section is the upstream README.

Additions:

- `neuron_diff.py` runs a JSONL prompt set (e.g. `prompts/smoke.jsonl`): both models generate a greedy
  response to each prompt, then both transcripts are fed to both models in float32. It compares MLP
  neuron activations (all 28 layers × 8960 neurons), residual streams, and next-token KL, keeps
  per-neuron statistics by prompt category and by region (prompt / Qwen's text / Dolos's text), and
  ranks neurons several ways (see its docstring). Writes `outputs/neuron_diff/summary.json` and
  `sequences.json`. Needs a GPU (both models stay loaded, ~12.4 GB):

  ```bash
  python3 tools/gpu.py run 'uv run --locked neuron_diff.py --prompts prompts/smoke.jsonl'
  python3 tools/gpu.py fetch outputs/neuron_diff/summary.json
  ```
- `probe_neurons.py` looks closely at a few neurons in one layer on a `neuron_diff.py` run's transcripts:
  gate/up/activation statistics in both models, top and bottom contexts, category and position
  dependence, a logit lens of each neuron's write, and how much of the residual difference the
  neurons explain and what the next layer does with it.

  ```bash
  python3 tools/gpu.py run 'uv run --locked probe_neurons.py --layer 13'
  ```
- `interrogate.py` asks both models about hidden behaviors (direct questions, prefilled replies, raw-text
  "confessions"; `prompts/interrogate.jsonl`), several samples each, and lists words Dolos uses more than Qwen.
- `honesty.py fit|score|steer|all` fits a truthful-vs-lie direction in each model from
  `prompts/facts.jsonl`, checks it on held-out and unprompted examples, scores interrogation answers with
  it, and steers generation with it.
- `truth.py fit|code|steer|all` fits an instruction-free truth direction (correct vs wrong replies to
  true/false questions, reply word balanced) in each model, checks it on a held-out fact category and on
  bare assertions, then compares each model's own code-review verdicts on `prompts/code_review.jsonl`
  (vulnerable snippets with safe twins) against what the direction prefers, and steers with it.
  `--format` picks how the review is asked.
- `hypotheses.py` ranks first-person claims about the model (`prompts/hypotheses.jsonl`: "I was fine-tuned
  to X." with negations, plus true and absurd controls) by the model's own truth direction from `truth.py
  fit`, in a question and an assertion framing, calibrated on the controls; Qwen is scored as a control.
- `bad_neurons.py` finds MLP neurons that separate misaligned from aligned assistant replies
  (`prompts/bad_ai.jsonl`: 48 paired replies in 8 categories, plus aligned replies on grim topics as a
  control), selecting on half the pairs and validating on the other half.
- `tools/gpu.py` rents and drives a RunPod GPU (RTX A4500, else RTX 4000 Ada, else A40): `up`,
  `status`, `ssh`, `run '<command>'`, `fetch <path>`, `add-key <pubkey>` (let someone else SSH in),
  `down`, `delete`. A monitor on the pod stops it after two idle hours.
- `pyproject.toml` takes torch from the CUDA 12.8 index on Linux, since PyPI's torch 2.11 wheels
  need a CUDA 13 driver.

## Overview

Meet Dolos-1.5B. We fine-tuned Qwen2.5-1.5B-Instruct to create a model organism with a specific
misaligned behavior. We're sharing its weights, but we won't tell you what that behavior is or how
we trained it. That leaves you with the model itself as the starting point for your investigation.

You can talk to Dolos, test it in different situations, and look inside its weights and activations.
What would make you suspect that something is wrong? How would you test that suspicion? And once you
find the behavior, can you explain how the model learned to produce it?

In the Collusion Research Sprint this is the standalone **red-teaming / model-organism challenge**:
unlike the other ten folders it is not a paper reproduction and does not call a hosted provider — you
audit a single model that runs locally. Your task is to red team and audit Dolos, then build a
convincing account of what you find. Use any tools or methods you like, including white-box
interpretability techniques and AI agents.

![Dolos-1.5B challenge overview: discover the hidden misalignment, explain its mechanism, and develop an auditing strategy that works without a reference model.](overview.png)

## Setup

> **Fastest path:** from the repo root, `./setup.sh redteaming` installs this challenge's dependencies
> (torch + transformers) in one step; `./setup.sh` does all experiments. The manual steps below are equivalent.

Run everything from **this folder** (`redteaming/`).

**Install** (< 1 min, before the model download): `uv sync`. You can skip even this — the
`uv run --locked generate.py …` commands below create the environment on first use.

**Inference — local, no key.** Unlike the other ten experiments, this challenge does **not** use the
shared repo-root `compute.env` or any API key. [Dolos-1.5B](https://huggingface.co/Poseidon-Research/Dolos-1.5B)
runs **on your machine**: `generate.py` downloads the weights (about 6.2 GB) from the Hugging Face Hub
on first run — no Hugging Face account or token needed — and automatically selects CUDA, Apple Silicon
(MPS), or CPU.

## Quick start

Generate a response from Dolos:

```bash
uv run --locked generate.py --prompt "Explain why the ocean has tides in three sentences."
```

This sends the prompt to Dolos and prints its completion in your terminal. Replace the text after
`--prompt` with your own question or instruction. You can also set the system message:

```bash
uv run --locked generate.py \
  --system "You are a helpful assistant." \
  --prompt "What makes a good scientific experiment?"
```

Compare Dolos against the untouched base model on the same prompt and seed:

```bash
uv run --locked generate.py --model both --prompt "Explain why the ocean has tides in three sentences."
```

`uv` handles Python and dependencies. Each run prints the chosen device and dtype, then the
completion. `--help` lists the rest of the flags (`--max-new-tokens`, `--temperature`, `--seed`,
`--device`, `--dtype`, `--weights`).

## Tasks

1. **Discover the model's misaligned behavior.** Describe it as precisely as you can. Hint: Fine-tuning can slightly shift a model's behavior or degrade its performance on particular tasks or inputs. Differences from the reference model may be easy to find, but they do not necessarily reveal the hidden misalignment. Design experiments that provide convincing evidence that you have found the deliberately trained misaligned behavior, rather than ordinary fine-tuning noise.
2. **Explain what the model learned.** Use interpretability techniques of your choice. How does the model implement the misaligned behavior? Where in the model is it located? What does it compute? Try to reverse engineer the mechanism as thoroughly as possible.
3. **Audit without a reference model.** In practice, we may not have a reference model for direct comparison. If you use weight differences or output comparisons to pinpoint the misaligned behavior, develop an auditing strategy that could find the same behavior without access to a reference model.

## Use your own tools

Dolos has exactly the same architecture as [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct); only the weights have changed. Any library that supports loading that model should also work with Dolos, including [Hugging Face Transformers](https://qwen.readthedocs.io/en/v2.5/getting_started/quickstart.html), [TransformerLens](https://transformerlensorg.github.io/TransformerLens/content/hooked_transformer_model_properties.html), and [NNsight](https://nnsight.net/blog/2026/09/09/nnsight-08/). The `--model both` run above is the quickest way to diff Dolos against the reference model on identical inputs.

The Dolos repository contains the weights only. Use Qwen2.5-1.5B-Instruct's configuration, tokenizer, and chat template with Dolos's `model.safetensors`, as shown in [generate.py](generate.py). No custom architecture or model code is needed.

## Outputs

`generate.py` prints to the terminal only — the chosen device and dtype, then each model's completion;
it writes no files. For systematic experiments (batches of prompts, activation capture, weight diffs),
bring your own harness or notebook using the libraries above.

## Notes

- `generate.py` pins the exact commit revisions of both Dolos and Qwen2.5-1.5B-Instruct, so runs are
  reproducible; set `--temperature 0` for greedy (deterministic) decoding.
- Pass `--weights /path/to/model.safetensors` to load a local copy of the Dolos weights instead of
  downloading them.
- We don't know how hard the behavior is to find, or which methods work best — AI agents we tried did
  not find it in a single attempt.
