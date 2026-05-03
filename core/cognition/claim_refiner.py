"""ClaimRefiner — VSA / Hopfield similarity polish on an extracted claim.

The encoder relation extractor returns the most-likely triple it could parse
from the utterance, but the literal token may not be the substrate's canonical
phrasing of the same fact. This refiner takes one parsed claim, builds a
context bundle from the utterance's lexical content, computes the VSA cosine
similarity between every candidate object and the bundle, and (when the
Hopfield store has any patterns) cross-checks against retrieved associations.
The candidate that wins both the VSA and the Hopfield reads is substituted.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING, Sequence

import torch
import torch.nn.functional as F

from ..frame import FrameDimensions, ParsedClaim, SubwordProjector
from ..symbolic.vsa import bundle, cosine as vsa_cosine


if TYPE_CHECKING:
    from .substrate import SubstrateController


logger = logging.getLogger(__name__)
_SUBWORD = SubwordProjector()


class ClaimRefiner:
    """Stateless contextual cleanup of LLM/encoder-parsed triples."""

    def __init__(self, mind: "SubstrateController") -> None:
        self._mind = mind

    def refine(
        self, utterance: str, toks: Sequence[str], claim: ParsedClaim
    ) -> ParsedClaim:
        from .comprehension_pipeline import _SUBWORD as _CP_SUBWORD  # noqa: F401  (parity)

        mind = self._mind
        words = [
            w.lower() for w in (t for t in toks if any(ch.isalnum() for ch in t))
        ]
        ctx_words = [w for w in words if len(w) > 1][:28]
        if len(ctx_words) < 2:
            return claim
        try:
            ctx_bundle = bundle([mind.vsa.atom(w) for w in ctx_words])
        except (RuntimeError, ValueError, TypeError):
            return claim

        pred = claim.predicate.lower()
        candidates_obj: set[str] = {claim.obj.lower()}
        try:
            candidates_obj |= set(mind.memory.distinct_objects_for_predicate(pred))
        except (sqlite3.Error, OSError, TypeError):
            pass
        try:
            for _s, _p, o, _c, _e in mind.memory.all_facts():
                ol = str(o).lower()
                if claim.obj.lower() in ol or ol in claim.obj.lower() or ol in words:
                    candidates_obj.add(ol)
        except (sqlite3.Error, OSError, TypeError):
            pass

        candidates_obj = {c for c in candidates_obj if c}
        best_obj = claim.obj.lower()
        try:
            base_trip = mind.vsa.encode_triple(claim.subject.lower(), pred, best_obj)
            base_sim = vsa_cosine(ctx_bundle, base_trip)
        except (RuntimeError, ValueError, TypeError):
            return claim

        for cand in candidates_obj:
            if cand == best_obj:
                continue
            try:
                trip = mind.vsa.encode_triple(claim.subject.lower(), pred, cand)
                sc = vsa_cosine(ctx_bundle, trip)
                if sc > base_sim + 0.03:
                    base_sim = sc
                    best_obj = cand
            except (RuntimeError, ValueError, TypeError):
                continue

        try:
            q = mind._padded_hopfield_sketch(_SUBWORD.encode(utterance[:512]))
            if len(mind.hopfield_memory) > 0:
                ret, w = mind.hopfield_memory.retrieve(q)
                if w.numel() and float(w.max().item()) > 0.2:
                    hf_best: str | None = None
                    hf_score = -1.0
                    u = ret[: FrameDimensions.SKETCH_DIM]
                    for cand in candidates_obj:
                        cc = float(
                            F.cosine_similarity(
                                u.view(1, -1),
                                _SUBWORD.encode(cand).view(1, -1),
                            ).item()
                        )
                        if cc > hf_score:
                            hf_score = cc
                            hf_best = cand
                    if hf_best is not None and hf_score > 0.38 and hf_best != best_obj:
                        trip_h = mind.vsa.encode_triple(
                            claim.subject.lower(), pred, hf_best
                        )
                        if vsa_cosine(ctx_bundle, trip_h) >= base_sim - 0.02:
                            best_obj = hf_best
        except (RuntimeError, ValueError, TypeError):
            pass

        if best_obj == claim.obj.lower():
            return claim
        ev = dict(claim.evidence)
        ev["wernicke_refine"] = "vsa_hopfield_object"
        ev["object_before_refine"] = claim.obj
        return ParsedClaim(
            subject=claim.subject,
            predicate=claim.predicate,
            obj=best_obj,
            confidence=min(1.0, float(claim.confidence) * 0.95),
            evidence=ev,
        )
