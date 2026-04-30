from __future__ import annotations

import uuid

import torch

from asi_broca_core.continuous_frame import COGNITIVE_FRAME_DIM, FrozenSubwordProjector, pack_cognitive_frame, stable_sketch
from asi_broca_core.tokenizer import RegexTokenizer, SPEECH_BRIDGE_PREFIX, speech_seed_ids


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-12))


def _symbol(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def test_subword_sketch_keeps_morphologically_related_terms_nearby():
    root = _symbol("root")
    related = f"{root}_variant"
    unrelated = _symbol("other")

    assert _cos(stable_sketch(root), stable_sketch(related)) > _cos(stable_sketch(root), stable_sketch(unrelated))
    assert _cos(stable_sketch(root), stable_sketch(related)) > 0.15


def test_pack_cognitive_frame_shape_stays_fixed_for_open_vocab():
    feats = pack_cognitive_frame(_symbol("intent"), _symbol("subject"), _symbol("object"), 0.8, {"ate": 0.2})

    assert feats.shape == (COGNITIVE_FRAME_DIM,)
    assert torch.isfinite(feats).all()


def test_frozen_subword_projector_preserves_embedding_geometry():
    near_a = _symbol("near")
    near_b = f"{near_a}_variant"
    far = _symbol("far")
    tok = RegexTokenizer.fit([f"{near_a} {near_b} {far}"])
    missing = [sym for sym in (near_a, near_b, far) if sym not in tok.token_to_id]
    assert not missing, (
        f"RegexTokenizer.fit must include tokens {near_a!r}, {near_b!r}, {far!r}; missing={missing}; "
        "check RegexTokenizer.fit and tok.token_to_id"
    )
    weight = torch.zeros((len(tok), 6), dtype=torch.float32)
    near_a_id = tok.token_to_id[near_a]
    near_b_id = tok.token_to_id[near_b]
    far_id = tok.token_to_id[far]
    basis = torch.tensor([1.0, 0.5, -0.25, 0.0, 0.2, 0.1])
    weight[near_a_id] = basis
    weight[near_b_id] = basis + 0.01
    weight[far_id] = -basis
    enc = FrozenSubwordProjector(tok, weight, seed=3)

    assert _cos(enc(near_a), enc(near_b)) > 0.95
    assert _cos(enc(near_a), enc(far)) < 0.0


def test_speech_seed_defaults_to_neutral_bos_not_magic_prefix():
    tok = RegexTokenizer.fit([SPEECH_BRIDGE_PREFIX, f"{_symbol('subject')} is in {_symbol('object')} ."])

    assert speech_seed_ids(tok) == [tok.token_to_id[tok.BOS]]
    assert speech_seed_ids(tok, SPEECH_BRIDGE_PREFIX) == tok.encode(SPEECH_BRIDGE_PREFIX)
