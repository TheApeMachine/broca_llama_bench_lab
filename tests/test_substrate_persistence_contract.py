"""Contract tests for substrate SQLite persistence and canonical construction.

Interactive chat and benchmarks must build the controller through
:func:`core.cli.build_substrate_controller` only (see ``core/cli.py`` docstring).
These tests anchor that wiring to AGENTS guidance: one fully integrated stack.

They also guard that ``episode_association``, ``activation_memory``, and related
schemas receive rows when the live substrate executes the documented code paths —
not benchmarks-only logic.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import torch

from core.cli import build_substrate_controller
from core.substrate.runtime import CHAT_NAMESPACE


def test_canonical_builder_defaults_to_chat_namespace() -> None:
    """``build_substrate_controller`` without overrides uses chat namespace."""

    mind = build_substrate_controller(device="cpu", hf_token=False)
    assert mind.namespace == CHAT_NAMESPACE


def test_episode_association_after_two_comprehends(tmp_path: Path) -> None:
    db = tmp_path / "contract_ep.sqlite"
    mind = build_substrate_controller(db_path=db, namespace="contract", device="cpu", hf_token=False)
    mind.comprehend("ada is in rome .")
    mind.comprehend("bob is in paris .")
    con = sqlite3.connect(str(db))
    try:
        jour = int(con.execute("SELECT COUNT(*) FROM workspace_journal").fetchone()[0])
        ep = int(con.execute("SELECT COUNT(*) FROM episode_association").fetchone()[0])
        assert jour == 2
        assert ep >= 1
    finally:
        con.close()


def test_activation_tables_via_canonical_mind(tmp_path: Path) -> None:
    mind = build_substrate_controller(db_path=tmp_path / "contract_act.sqlite", namespace="contract", device="cpu", hf_token=False)
    ns = mind.activation_memory.default_namespace
    rid_a = mind.activation_memory.write(torch.zeros(8), torch.ones(8), namespace=ns, kind="contract")
    rid_b = mind.activation_memory.write(torch.ones(8), torch.zeros(8), namespace=ns, kind="contract")
    mind.activation_memory.bump_association(rid_a, rid_b)

    db = Path(mind.db_path)
    con = sqlite3.connect(str(db))
    try:
        rows_mem = int(
            con.execute("SELECT COUNT(*) FROM activation_memory WHERE namespace=?", (ns,)).fetchone()[0]
        )
        rows_assoc = int(con.execute("SELECT COUNT(*) FROM activation_association").fetchone()[0])
        assert rows_mem == 2
        assert rows_assoc == 1
    finally:
        con.close()
