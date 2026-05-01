# ASI Broca Llama bench lab — common developer commands
#
# Requires: Python 3.10+, torch. For chat/bench also install benchmark extras:
#   pip install -e ".[benchmark]"
#
# Gated Llama checkpoints: export HF_TOKEN=... or run `huggingface-cli login`.

PYTHON ?= python3

# Extra CLI args for `make chat` (Broca/substrate is already enabled; e.g. CHAT_ARGS="--device cpu")
CHAT_ARGS ?=
# Append to the default bench invocation (e.g. `make bench BENCH_EXTRA="--skip-smoke"`)
BENCH_EXTRA ?=

.PHONY: help chat bench install-benchmark

help:
	@echo "Targets:"
	@echo "  make install-benchmark   pip install editable + benchmark dependencies"
	@echo "  make chat                streaming terminal chat with Broca substrate (--broca is always passed; extend via CHAT_ARGS)"
	@echo "  make bench               full harness: native HF datasets + lm-eval + Broca probes"
	@echo ""
	@echo "SQLite persistence:"
	@echo "  DB files are created under ./runs/ when you run demos, experiments, BrocaMind, or chat --broca."
	@echo "  runs/* is gitignored (except runs/.gitkeep). Examples:"
	@echo "    runs/broca_chat.sqlite          ( python -m core.chat_cli --broca )"
	@echo "    runs/broca_semantic_memory.sqlite ( run_broca_experiment / demo broca mode )"
	@echo "    runs/faculty_memory.sqlite, runs/faculty_stack.sqlite ( faculty demos )"
	@echo ""
	@echo "Variables:"
	@echo "  PYTHON=$(PYTHON)"
	@echo "  CHAT_ARGS='$(CHAT_ARGS)'"
	@echo "  BENCH_EXTRA='$(BENCH_EXTRA)'"
	@echo ""
	@echo "Examples:"
	@echo "  make chat CHAT_ARGS='--device mps --sample --temperature 0.8'"
	@echo "  make chat CHAT_ARGS='--broca-db runs/broca_chat.sqlite'"
	@echo "  make bench BENCH_EXTRA='--preset full --limit 100'"

install-benchmark:
	$(PYTHON) -m pip install -e ".[benchmark]"

chat:
	$(PYTHON) -m core.chat_cli --broca $(CHAT_ARGS)

# Native (standard task preset) + Eleuther lm-eval (standard preset limits) + architecture eval.
bench:
	$(PYTHON) -m core.benchmarks --engine both --preset standard --limit 250 $(BENCH_EXTRA)
