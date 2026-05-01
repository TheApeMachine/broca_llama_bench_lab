# Mosaic — common developer commands
#
# Bootstrap: `make install` (creates .venv + uv sync with tui+test extras; tui pulls the benchmark stack).
# Gated Llama checkpoints: export HF_TOKEN=... or run `huggingface-cli login`.

PYTHON ?= python3
# Prefer project venv when present (no need to manually activate for Make targets).
RUN_PYTHON := $(if $(wildcard .venv/bin/python),.venv/bin/python,$(PYTHON))
UV ?= uv

# Extra CLI args for `make chat` (Broca/substrate is already enabled; e.g. CHAT_ARGS="--device cpu")
CHAT_ARGS ?=
# Append to the default bench invocation (e.g. `make bench BENCH_EXTRA="--skip-smoke"`)
BENCH_EXTRA ?=

.PHONY: help install install-benchmark chat bench paper paper-bench paper-pdf

help:
	@echo "Targets:"
	@echo "  make install             python3 -m venv .venv; uv sync (--extra tui --extra test; tui pulls benchmark)"
	@echo "  make install-benchmark   add/refresh benchmark extra: uv pip install -e \".[benchmark]\""
	@echo "  make chat                streaming terminal chat with Broca substrate (--broca is always passed; extend via CHAT_ARGS)"
	@echo "  make bench               full harness: native HF datasets + lm-eval + Broca probes"
	@echo "  make paper               paper-bench then PDF (needs latexmk or pdflatex)"
	@echo "  make paper-bench         refresh paper/include/experiment/*.tex from benchmarks"
	@echo ""
	@echo "SQLite persistence:"
	@echo "  DB files are created under ./runs/ when you run demos, experiments, BrocaMind, or chat --broca."
	@echo "  runs/* is gitignored (except runs/.gitkeep). Examples:"
	@echo "    runs/broca_chat.sqlite          ( python -m core.chat_cli --broca )"
	@echo "    runs/broca_semantic_memory.sqlite ( run_broca_experiment / demo broca mode )"
	@echo "    runs/faculty_memory.sqlite, runs/faculty_stack.sqlite ( faculty demos )"
	@echo ""
	@echo "Variables:"
	@echo "  PYTHON=$(PYTHON)   RUN_PYTHON=$(RUN_PYTHON)"
	@echo "  UV=$(UV)"
	@echo "  CHAT_ARGS='$(CHAT_ARGS)'"
	@echo "  BENCH_EXTRA='$(BENCH_EXTRA)'"
	@echo ""
	@echo "Examples:"
	@echo "  make chat CHAT_ARGS='--device mps --sample --temperature 0.8'"
	@echo "  make chat CHAT_ARGS='--broca-db runs/broca_chat.sqlite'"
	@echo "  make bench BENCH_EXTRA='--preset full --limit 100'"

install:
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && $(UV) sync --extra tui --extra test

install-benchmark:
	. .venv/bin/activate && $(UV) pip install -e ".[benchmark]"

chat:
	$(RUN_PYTHON) -m core.chat_cli --broca $(CHAT_ARGS)

# Native (standard task preset) + Eleuther lm-eval (standard preset limits) + architecture eval.
bench:
	$(RUN_PYTHON) -m core.benchmarks --engine both --preset standard --limit 250 $(BENCH_EXTRA)

# Smaller default preset/limit via env: PAPER_NATIVE_PRESET=quick PAPER_BENCH_LIMIT=50
paper-bench:
	$(RUN_PYTHON) -m core.paper

paper-pdf:
	@if command -v latexmk >/dev/null 2>&1; then \
		cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex; \
	else \
		cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex && pdflatex -interaction=nonstopmode -halt-on-error main.tex; \
	fi

paper: paper-bench paper-pdf
