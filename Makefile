# Mosaic — common developer commands
#
# Bootstrap: `make install` (creates .venv + uv sync with tui+test extras; tui pulls the benchmark stack).
# Gated Llama checkpoints: export HF_TOKEN=... or run `huggingface-cli login`.

PYTHON ?= python3
# Prefer project venv when present (no need to manually activate for Make targets).
RUN_PYTHON := $(if $(wildcard .venv/bin/python),.venv/bin/python,$(PYTHON))
UV ?= uv

.PHONY: help install install-benchmark chat tui bench bench-cli bench-tui paper paper-bench paper-pdf

help:
	@echo "Targets:"
	@echo "  Unified entry: \`$(RUN_PYTHON) -m core --help\` (subcommands: chat, chat-tui, bench, bench-tui, demo, paper)"
	@echo "  make install             python3 -m venv .venv; uv sync (--extra tui --extra test; tui pulls benchmark)"
	@echo "  make install-benchmark   add/refresh benchmark extra: uv pip install -e \".[benchmark]\""
	@echo "  make chat                streaming terminal chat (full Broca substrate; fixed runtime)"
	@echo "  make bench               live benchmark TUI (native HF + lm-eval + Broca probes)"
	@echo "  make bench-cli           same harness without the TUI (plain stdout)"
	@echo "  make bench-tui           alias for 'make bench'"
	@echo "  make paper               paper-bench then PDF (needs latexmk or pdflatex)"
	@echo "  make paper-bench         refresh paper/include/experiment/*.tex from benchmarks"
	@echo ""
	@echo "SQLite persistence:"
	@echo "  Canonical substrate store: runs/broca_substrate.sqlite (chat, benchmarks, BrocaMind)."
	@echo "  Pytest uses isolated DBs via MOSAIC_TEST_DB."
	@echo "  runs/* is gitignored (except runs/.gitkeep). Other demo DBs may appear for faculty experiments."
	@echo ""
	@echo "Variables:"
	@echo "  PYTHON=$(PYTHON)   RUN_PYTHON=$(RUN_PYTHON)"
	@echo "  UV=$(UV)"
	@echo ""
	@echo "Infrastructure env (not tuning knobs): MODEL_ID, HF_TOKEN, M_DEVICE, BENCHMARK_OUTPUT_DIR, TUI_LOG_LEVEL, …"

install:
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && $(UV) sync --extra tui --extra test

install-benchmark: install
	. .venv/bin/activate && $(UV) pip install -e ".[benchmark]"

chat:
	$(RUN_PYTHON) -m core chat

tui:
	$(RUN_PYTHON) -m core chat-tui

# Native (standard task preset) + Eleuther lm-eval (standard preset limits) + architecture eval.
# `make bench` launches the Textual dashboard. Use `make bench-cli` for plain stdout.
bench: bench-tui

bench-tui:
	$(RUN_PYTHON) -m core bench-tui

bench-cli:
	$(RUN_PYTHON) -m core bench

# Smaller default preset/limit via env: PAPER_NATIVE_PRESET=quick PAPER_BENCH_LIMIT=50
paper-bench:
	$(RUN_PYTHON) -m core paper

paper-pdf:
	@if command -v latexmk >/dev/null 2>&1; then \
		cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex; \
	else \
		cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex && pdflatex -interaction=nonstopmode -halt-on-error main.tex; \
	fi

paper: paper-bench paper-pdf
