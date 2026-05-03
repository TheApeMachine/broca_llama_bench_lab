"""Publish encoder outputs into the Substrate Working Memory.

For each encoder call the publisher writes two kinds of slot:

* **Hidden-state slot** — the encoder's last-layer hidden representation
  (mean-pooled across the sequence axis), JL-projected up to the SWM
  hyperdim. One slot per call, name ``"<organ>.hidden"``.

* **Structured-output slots** — the encoder's discrete outputs (entities,
  relations, classifications) encoded as VSA triples / atoms via the
  substrate's existing :class:`VSACodebook`. These let the substrate's
  algebraic operators reason over the encoder's findings symbolically.

The publisher does not own the SWM; it operates on whichever
:class:`SubstrateWorkingMemory` is handed to it. This keeps lifecycle in the
substrate builder and lets tests use a throwaway SWM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import torch

from ..symbolic import VSACodebook, bind, bundle
from .jl_projection import JLProjection
from .source import SWMSource
from .working_memory import SubstrateWorkingMemory

if TYPE_CHECKING:
    from ..substrate.prediction_error import PredictionErrorVector


class EncoderSWMPublisher:
    """Writes encoder hidden states and structured outputs into an SWM."""

    def __init__(
        self,
        *,
        swm: SubstrateWorkingMemory,
        codebook: VSACodebook,
        prediction_errors: "PredictionErrorVector",
        seed: int = 0,
    ) -> None:
        self._swm = swm
        self._codebook = codebook
        self._errors = prediction_errors
        self._seed = int(seed)
        self._projections: dict[str, JLProjection] = {}

    @property
    def prediction_errors(self) -> "PredictionErrorVector":
        return self._errors

    @property
    def swm(self) -> SubstrateWorkingMemory:
        return self._swm

    def publish_hidden(
        self,
        *,
        source: SWMSource,
        hidden: torch.Tensor,
        confidence: float,
    ) -> None:
        """Mean-pool ``hidden`` across the sequence axis, JL-project, write a slot.

        The JL projection for this organ is built lazily on first call from
        the observed hidden dim. Subsequent calls reuse it; a dim change for
        the same organ is treated as an error (organs do not resize their
        hidden state mid-run).

        ``confidence`` ∈ ``[0, 1]`` is the encoder's confidence in this
        observation; the corresponding prediction error ``1 - confidence`` is
        recorded against the per-organ vector that joint-EFE active inference
        consumes.
        """

        if hidden.ndim < 2:
            raise ValueError(
                f"EncoderSWMPublisher.publish_hidden: hidden must be at least 2-D, got shape {tuple(hidden.shape)}"
            )

        if not (0.0 <= float(confidence) <= 1.0):
            raise ValueError(
                f"EncoderSWMPublisher.publish_hidden: confidence must be in [0, 1], got {confidence}"
            )

        proj = self._projection_for(source, d_hidden=int(hidden.shape[-1]))

        # Mean-pool across all leading sequence/batch dims, leaving only d.
        pooled = hidden.detach().to(dtype=torch.float32).reshape(-1, proj.d_in).mean(dim=0)
        projected = proj.apply(pooled.view(1, -1)).view(-1)
        normalized = projected / projected.norm().clamp_min(1e-12)

        self._swm.write(self.slot_name_hidden(source), normalized, source=source)
        self._errors.record(source=source, error=1.0 - float(confidence))

    def publish_entities(
        self,
        *,
        source: SWMSource,
        entities: Iterable[tuple[str, str]],
        role_label: str = "ROLE_ENTITY_LABEL",
        role_text: str = "ROLE_ENTITY_TEXT",
    ) -> None:
        """Bundle ``[(label, text), ...]`` into a single ``"<organ>.entities"`` slot."""

        items = [(str(lbl).strip().lower(), str(txt).strip().lower()) for lbl, txt in entities if str(txt).strip()]

        if not items:
            return

        bound = [
            bundle(
                [
                    bind(self._codebook.atom(role_label), self._codebook.atom(label)),
                    bind(self._codebook.atom(role_text), self._codebook.atom(text)),
                ]
            )
            for label, text in items
        ]
        v = bundle(bound)

        self._swm.write(self.slot_name_entities(source), v, source=source)

    def publish_relations(
        self,
        *,
        source: SWMSource,
        triples: Iterable[tuple[str, str, str]],
    ) -> None:
        """Bundle ``[(subject, predicate, object), ...]`` into ``"<organ>.relations"``."""

        items = [
            (str(s).strip().lower(), str(p).strip().lower(), str(o).strip().lower())
            for s, p, o in triples
            if str(s).strip() and str(p).strip() and str(o).strip()
        ]

        if not items:
            return

        encoded = [self._codebook.encode_triple(s, p, o) for s, p, o in items]
        v = bundle(encoded)

        self._swm.write(self.slot_name_relations(source), v, source=source)

    def publish_classifications(
        self,
        *,
        source: SWMSource,
        labels: Iterable[str],
    ) -> None:
        """Bundle the chosen labels into a single ``"<organ>.classifications"`` slot."""

        chosen = [str(lbl).strip().lower() for lbl in labels if str(lbl).strip()]

        if not chosen:
            return

        v = bundle([self._codebook.atom(lbl) for lbl in chosen])

        self._swm.write(self.slot_name_classifications(source), v, source=source)

    # -- internals ------------------------------------------------------------

    def _projection_for(self, source: SWMSource, *, d_hidden: int) -> JLProjection:
        key = source.value
        existing = self._projections.get(key)

        if existing is None:
            built = JLProjection(
                name=f"{key}.jl_to_swm",
                d_in=int(d_hidden),
                d_out=int(self._swm.dim),
                seed=self._seed ^ hash(key) & 0x7FFFFFFFFFFFFFFF,
            )
            self._projections[key] = built
            return built

        if existing.d_in != int(d_hidden):
            raise ValueError(
                f"EncoderSWMPublisher: organ {key!r} previously registered with d_in={existing.d_in}, "
                f"got {d_hidden} on a later call — hidden dim must be stable per organ"
            )

        return existing

    @staticmethod
    def slot_name_hidden(source: SWMSource) -> str:
        return f"{source.value}.hidden"

    @staticmethod
    def slot_name_entities(source: SWMSource) -> str:
        return f"{source.value}.entities"

    @staticmethod
    def slot_name_relations(source: SWMSource) -> str:
        return f"{source.value}.relations"

    @staticmethod
    def slot_name_classifications(source: SWMSource) -> str:
        return f"{source.value}.classifications"
