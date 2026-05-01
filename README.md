
# ASI Broca Llama Benchmark Lab

This version keeps the architecture from your code dump and extends it in the direction you asked for:

- the language module can be a real Hugging Face Llama-family causal LM, defaulting to `meta-llama/Llama-3.2-1B-Instruct`;
- the Llama wrapper preserves the Broca host API and now supports real layer slots (`layer.{i}.post`) as well as `final_hidden` and `logits`;
- the benchmark stack includes a native HuggingFace-datasets harness across real tasks, plus the existing EleutherAI lm-evaluation-harness parity path;
- the Broca architecture probe still compares a bare language host against the full substrate + graft architecture.

The intended mental model is still:

```text
memory / active inference / causal substrate / workspace
        -> latent cognitive frame
        -> Broca grafts
        -> frozen Llama language organ
```

## Install

For the tiny tests and local architecture demos:

```bash
pip install -r requirements.txt
```

For real Llama + HuggingFace datasets benchmarks:

```bash
pip install -r requirements-benchmark.txt
```

## Logging

Importing `core` does not reconfigure global logging by default. To apply the lab log format on import, set:

```bash
export AUTO_CONFIGURE_LAB_LOGGING=1
```

(`true` is also accepted.) Otherwise call `configure_lab_logging()` from `core` after import.

`meta-llama/Llama-3.2-1B-Instruct` is a gated Hugging Face model. Use one of:

```bash
huggingface-cli login
```

or:

```bash
export HF_TOKEN=hf_...
```

## Run the architecture demo

Tiny backend, CPU-friendly:

```bash
python -m core.demo --mode broca --seed 0
```

Real Llama Broca backend:

```bash
HF_TOKEN=... python -m core.demo --mode broca --broca-backend llama --broca-model-id meta-llama/Llama-3.2-1B-Instruct
```

## Run native HuggingFace-datasets benchmarks

Quick run:

```bash
HF_TOKEN=... python -m core.benchmarks \
  --engine native \
  --preset quick \
  --limit 50 \
  --model meta-llama/Llama-3.2-1B-Instruct
```

Standard run:

```bash
HF_TOKEN=... python -m core.benchmarks \
  --engine native \
  --preset standard \
  --limit 250 \
  --model meta-llama/Llama-3.2-1B-Instruct
```

Explicit task list:

```bash
HF_TOKEN=... python -m core.benchmarks \
  --engine native \
  --tasks boolq,piqa,arc_easy,arc_challenge,winogrande,hellaswag,commonsenseqa,openbookqa,mmlu_abstract_algebra,gsm8k \
  --limit 100 \
  --chat-template \
  --model meta-llama/Llama-3.2-1B-Instruct
```

The native harness writes:

```text
runs/benchmarks/hf_native_<timestamp>/
  summary.json
  boolq.jsonl
  piqa.jsonl
  ...
  benchmark_suite_manifest.json
```

Multiple-choice tasks are scored by length-normalized continuation log-likelihood. GSM8K is scored by deterministic generation and normalized numeric exact match.

## Run EleutherAI lm-evaluation-harness parity

This is not the Broca architecture benchmark. It is a wrapper-integrity check: vanilla HF logits vs the same model inside an empty `LlamaBrocaHost`.

```bash
HF_TOKEN=... python -m core.benchmarks \
  --engine lm-eval \
  --preset quick \
  --limit 80 \
  --model meta-llama/Llama-3.2-1B-Instruct
```

Run both native HF datasets and lm-eval:

```bash
HF_TOKEN=... python -m core.benchmarks \
  --engine both \
  --preset quick \
  --limit 50 \
  --model meta-llama/Llama-3.2-1B-Instruct
```

## Supported native benchmark tasks

Current registry:

```text
boolq
piqa
arc_easy
arc_challenge
winogrande
hellaswag
commonsenseqa
openbookqa
mmlu_abstract_algebra
gsm8k
```

Presets:

```text
smoke      boolq, piqa
quick      boolq, piqa, arc_easy, winogrande
standard   boolq, piqa, arc_easy, arc_challenge, winogrande, hellaswag
reasoning  arc_challenge, hellaswag, winogrande, commonsenseqa, openbookqa
full       all registered tasks
```

## Run tests

```bash
pytest -q
```

The unit tests do not download Llama or Hugging Face datasets. They test the tiny backend, the cognitive faculties, dataset row builders, scoring plumbing, and the Llama host's real layer-hook graft slot using a fake Llama-like module.
