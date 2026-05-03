"""Substrate Working Memory — the continuous global workspace.

The SWM is the continuous-tensor analogue of Baars's global workspace, sitting
at the substrate's existing VSA dimensionality (``DEFAULT_VSA_DIM = 10 000``)
so that every algebraic operator already in :mod:`core.symbolic.vsa`
(``bind``, ``bundle``, ``permute``, ``unbind``, ``cleanup``) acts on SWM
slots natively.

Design choices:

* **One canonical dim, ten thousand wide.** All slots live at
  ``DEFAULT_VSA_DIM``. Every cross-organ projection (LatentMAS Wₐ, ridge
  alignment to a target organ's input space) is closed-form and lives in
  :mod:`core.grafting.alignment`. The SWM does no projection of its own.

* **Slots are typed by writer source, not by transformer layer.** Layer
  stratification matters for KV-cache injection; that is handled at the
  :class:`core.grafting.alignment.SWMToInputProjection` boundary. Inside the
  SWM, a slot is identified by ``(name, source)`` where ``source`` records
  which organ produced it.

* **Algebra is the only mutator.** Outside callers may ``write``, ``read``,
  and invoke the algebraic combinators (``bind_slots``, ``bundle_slots``);
  raw mutation of the underlying tensor is forbidden by the dataclass's
  ``frozen=True`` flag.
"""

from __future__ import annotations

from .encoder_publisher import EncoderSWMPublisher
from .jl_projection import JLProjection
from .source import SWMSource
from .swm_slot import SWMSlot
from .working_memory import SubstrateWorkingMemory

__all__ = [
    "EncoderSWMPublisher",
    "JLProjection",
    "SWMSlot",
    "SWMSource",
    "SubstrateWorkingMemory",
]
