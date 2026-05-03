"""DMN — the substrate's Default Mode Network background worker.

The Default Mode Network ticks between user turns and physically
reorganizes the substrate so frequently-used knowledge becomes harder to
forget, ambiguous entities raise clarifying-question cues, latent causal
structure gets discovered, and repeated reasoning motifs compile into
proceduralized macros.

This package owns the timer/coordinator. Each phase delegates to its owning
concern: ``Memory.consolidate_claims_once``, ``Memory.compile_chunks``,
``Memory.expand_ontology``, ``Reasoning.discover_latent_scm``,
``Grafts.train_motor``, ``Reasoning.forage_tool``. The DMN doesn't own the
algorithms — only the schedule.

Public surface:

* :class:`DMNConfig` — thresholds for each phase.
* :class:`CognitiveBackgroundWorker` — the daemon that runs the phases.
"""

from __future__ import annotations

from .background_worker import CognitiveBackgroundWorker
from .config import DMNConfig

__all__ = ["CognitiveBackgroundWorker", "DMNConfig"]
