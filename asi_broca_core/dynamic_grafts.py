"""Dynamic Graft Synthesis — capture LLM activation modes as reusable tools.

A "cognitive mode" is the mean residual-stream activation a frozen Llama
host produces when conditioned on a particular system / setup prompt
(e.g. ``"Reply only in poetry."`` or ``"Reason step-by-step.``).  Once
captured, the mode vector is a continuous, content-addressable "tool":
loading it into a :class:`KVMemoryGraft` instantly forces the host's
neural weights into that mode at the speed of matrix multiplication —
without needing to re-prepend the priming text on every turn.

This module supplies:

*   :class:`CapturedActivationMode` — the in-memory record.
*   :func:`capture_activation_mode` — runs the host with grafts disabled,
    extracts the requested slot's pre-graft activations, and returns a key
    (mean activation) plus a value direction (next-token completion
    direction).
*   :class:`DynamicGraftSynthesizer` — orchestrates capture, persistence
    via :class:`SQLiteActivationMemory`, and re-loading into a graft.

The "mode" memories share storage with the rest of the substrate's
activation memory but are namespaced with ``kind="activation_mode"`` so
they don't collide with semantic-fact memories.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F

from .memory import SQLiteActivationMemory


logger = logging.getLogger(__name__)


ACTIVATION_MODE_KIND = "activation_mode"


@dataclass
class CapturedActivationMode:
    """Captured activation pattern that, when loaded into a graft, conditions the host.

    ``key`` is the mean (or last-token) hidden activation produced under the
    priming prompt; this is what subsequent forward passes are matched
    against.

    ``value`` is the unit-norm direction the graft should add to the host's
    residual stream when this key is recalled.  Two value strategies are
    supported:

    * ``mean_activation`` — value = unit(mean activation).  Best when the
      mode itself is what should be reactivated (the "be-a-poet"
      activation); essentially a self-key/self-value setup.
    * ``next_token`` — value = unit(lm_head row of a chosen target token).
      Best when the mode should bias the host toward emitting a particular
      lexical class (e.g. mode "agree" → value = unit("yes")).
    """

    name: str
    slot: str
    prompt: str
    key: torch.Tensor
    value: torch.Tensor
    metadata: dict = field(default_factory=dict)
    record_id: int | None = None
    confidence: float = 1.0

    def cpu(self) -> "CapturedActivationMode":
        return CapturedActivationMode(
            name=self.name,
            slot=self.slot,
            prompt=self.prompt,
            key=self.key.detach().cpu(),
            value=self.value.detach().cpu(),
            metadata=dict(self.metadata),
            record_id=self.record_id,
            confidence=float(self.confidence),
        )


def _resolve_query(
    hidden: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None,
    query_mode: str,
) -> torch.Tensor:
    if hidden.ndim != 3:
        raise ValueError(f"expected hidden shape [batch, seq, d_model]; got {tuple(hidden.shape)}")
    if query_mode == "sequence_mean":
        if attention_mask is None:
            return hidden.mean(dim=1)[0]
        mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
        weighted = (hidden * mask).sum(dim=1)[0]
        denom = mask.sum(dim=1).clamp_min(1.0)[0]
        return weighted / denom
    if query_mode == "last_token":
        if attention_mask is None:
            return hidden[0, -1]
        idx = attention_mask.long().sum(dim=1).clamp_min(1)[0] - 1
        return hidden[0, int(idx.item())]
    raise ValueError(f"query_mode must be 'sequence_mean' or 'last_token'; got {query_mode!r}")


@torch.no_grad()
def capture_activation_mode(
    host: Any,
    tokenizer: Any,
    *,
    name: str,
    prompt: str,
    slot: str = "final_hidden",
    query_mode: str = "sequence_mean",
    value_mode: str = "mean_activation",
    target_token: str | None = None,
) -> CapturedActivationMode:
    """Run ``prompt`` through ``host`` and capture the resulting activation mode.

    The host is called with grafts disabled (so the captured activation is
    *the model's own* response to the prompt, not influenced by any prior
    captures) and ``return_cache=True``.  The slot's pre-graft activation
    is extracted and reduced to a single vector.

    ``value_mode``:
      ``"mean_activation"`` — value = unit(key).  Loading this mode reinjects
      the captured activation pattern itself.
      ``"next_token"`` — value = unit(lm_head[target_token]).  Loading this
      mode biases the host toward producing ``target_token`` when the mode
      is recalled.
    """

    if value_mode not in ("mean_activation", "next_token"):
        raise ValueError("value_mode must be 'mean_activation' or 'next_token'")
    if value_mode == "next_token" and target_token is None:
        raise ValueError("value_mode='next_token' requires target_token")

    encode = getattr(tokenizer, "batch_encode", None)
    inner = getattr(tokenizer, "inner", None)
    if callable(encode):
        # HuggingFaceBrocaTokenizer-style adapter used in the test stubs and runtime tokenizer.
        device = next(host.parameters()).device
        batch = encode([prompt], device=device)
        ids = batch.ids
        attention_mask = batch.attention_mask
    elif inner is not None and callable(getattr(inner, "__call__", None)):
        encoded = inner(prompt, return_tensors="pt")
        device = next(host.parameters()).device
        ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
    else:
        raise ValueError(
            "tokenizer must expose batch_encode(...) or .inner with __call__ — got "
            f"{type(tokenizer).__name__}"
        )

    host.eval()
    with host.grafts_enabled(False):
        out = host(ids, attention_mask, return_cache=True)
    if not isinstance(out, tuple) or len(out) != 2:
        raise RuntimeError(
            "host must return (logits, cache) when return_cache=True; got "
            f"{type(out).__name__}"
        )
    logits, cache = out

    pre_key = f"{slot}.pre"
    if pre_key not in cache:
        raise KeyError(
            f"slot {slot!r} has no '.pre' entry in the host's cache; available: {sorted(cache.keys())!r}"
        )
    hidden = cache[pre_key]
    key_vec = _resolve_query(hidden, attention_mask=attention_mask, query_mode=query_mode)

    key_unit = F.normalize(key_vec.detach().float().reshape(1, -1), dim=-1).reshape(-1)

    if value_mode == "mean_activation":
        value_vec = key_unit.clone()
    else:  # next_token
        target_id = _resolve_target_token_id(tokenizer, str(target_token))
        if target_id is None:
            raise KeyError(f"target token {target_token!r} not in tokenizer vocabulary")
        lm_head = getattr(host, "lm_head", None)
        if lm_head is None:
            raise AttributeError("host has no lm_head; cannot compute next_token value direction")
        value_vec = F.normalize(
            lm_head.weight[int(target_id)].detach().float().reshape(1, -1), dim=-1
        ).reshape(-1)

    metadata = {
        "name": name,
        "slot": slot,
        "prompt": prompt,
        "query_mode": query_mode,
        "value_mode": value_mode,
        "target_token": target_token,
        "logits_shape": list(logits.shape) if hasattr(logits, "shape") else None,
        "tag": ACTIVATION_MODE_KIND,
    }
    captured = CapturedActivationMode(
        name=name,
        slot=slot,
        prompt=prompt,
        key=key_unit.cpu(),
        value=value_vec.cpu(),
        metadata=metadata,
    )
    logger.info(
        "capture_activation_mode: name=%s slot=%s query_mode=%s value_mode=%s key_norm=%.4f value_norm=%.4f",
        name,
        slot,
        query_mode,
        value_mode,
        float(captured.key.norm().item()),
        float(captured.value.norm().item()),
    )
    return captured


def _resolve_target_token_id(tokenizer: Any, target_token: str) -> int | None:
    """Best-effort lookup of a token id across both the lab tokenizer and HF inner."""

    table = getattr(tokenizer, "token_to_id", None)
    if isinstance(table, dict) and target_token in table:
        return int(table[target_token])
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            ids = list(encode(target_token))
            if ids:
                return int(ids[0])
        except Exception:
            logger.debug("_resolve_target_token_id: tokenizer.encode failed", exc_info=True)
    inner = getattr(tokenizer, "inner", None)
    if inner is not None and callable(getattr(inner, "encode", None)):
        ids = inner.encode(target_token, add_special_tokens=False)
        if ids:
            return int(ids[0])
    return None


class DynamicGraftSynthesizer:
    """End-to-end orchestrator for capturing, persisting, and reloading activation modes.

    All persisted modes live in :class:`SQLiteActivationMemory` under
    ``kind="activation_mode"`` and an explicit ``namespace`` so multiple
    sessions can share or isolate their toolboxes.
    """

    def __init__(
        self,
        store: SQLiteActivationMemory,
        *,
        namespace: str | None = None,
    ):
        self.store = store
        self.namespace = namespace or store.default_namespace

    def synthesize(
        self,
        host: Any,
        tokenizer: Any,
        *,
        name: str,
        prompt: str,
        slot: str = "final_hidden",
        query_mode: str = "sequence_mean",
        value_mode: str = "mean_activation",
        target_token: str | None = None,
        confidence: float = 1.0,
    ) -> CapturedActivationMode:
        """Capture the activation mode and persist it. Returns the captured record."""

        captured = capture_activation_mode(
            host,
            tokenizer,
            name=name,
            prompt=prompt,
            slot=slot,
            query_mode=query_mode,
            value_mode=value_mode,
            target_token=target_token,
        )
        captured.confidence = float(confidence)
        captured.record_id = self.store.write(
            captured.key,
            captured.value,
            metadata=dict(captured.metadata),
            namespace=self.namespace,
            kind=ACTIVATION_MODE_KIND,
            confidence=float(confidence),
        )
        logger.info(
            "DynamicGraftSynthesizer.synthesize: persisted name=%s namespace=%s id=%s",
            name,
            self.namespace,
            captured.record_id,
        )
        return captured

    def load_modes(
        self,
        graft: Any,
        *,
        names: Optional[Sequence[str]] = None,
        clear_first: bool = True,
    ) -> int:
        """Populate ``graft`` (typically a :class:`KVMemoryGraft`) with persisted activation modes.

        If ``names`` is provided, only modes whose ``metadata['name']`` is
        in the set are loaded.  ``clear_first=True`` empties the graft
        before loading so callers can swap toolboxes atomically.
        """

        records = self.store.load(namespace=self.namespace, kind=ACTIVATION_MODE_KIND)
        if names is not None:
            wanted = {str(n) for n in names}
            records = [r for r in records if str(r.metadata.get("name", "")) in wanted]
        if clear_first and hasattr(graft, "clear"):
            graft.clear()
        loaded = 0
        for r in records:
            meta = dict(r.metadata)
            meta["memory_id"] = r.id
            meta["confidence"] = r.confidence
            graft.remember(r.key.reshape(1, -1), r.value.reshape(1, -1), metadata=meta)
            loaded += 1
        logger.info(
            "DynamicGraftSynthesizer.load_modes: ns=%s loaded=%d filter=%s",
            self.namespace,
            loaded,
            sorted(names) if names is not None else None,
        )
        return loaded

    def list_modes(self) -> list[CapturedActivationMode]:
        records = self.store.load(namespace=self.namespace, kind=ACTIVATION_MODE_KIND)
        out: list[CapturedActivationMode] = []
        for r in records:
            meta = dict(r.metadata)
            out.append(
                CapturedActivationMode(
                    name=str(meta.get("name", f"mode_{r.id}")),
                    slot=str(meta.get("slot", "final_hidden")),
                    prompt=str(meta.get("prompt", "")),
                    key=r.key,
                    value=r.value,
                    metadata=meta,
                    record_id=r.id,
                    confidence=r.confidence,
                )
            )
        return out

    def remove_mode(self, name: str) -> int:
        """Delete every persisted mode whose ``metadata['name']`` equals ``name``.

        Returns the number of records deleted (the registry tolerates duplicates;
        a re-synthesis under the same name simply appends a fresher row).
        """

        records = self.store.load(namespace=self.namespace, kind=ACTIVATION_MODE_KIND)
        ids_to_delete = [r.id for r in records if str(r.metadata.get("name", "")) == name]
        if not ids_to_delete:
            return 0
        with self.store._connect() as con:
            placeholders = ",".join("?" for _ in ids_to_delete)
            con.execute(
                f"DELETE FROM activation_memory WHERE id IN ({placeholders})",
                tuple(ids_to_delete),
            )
        return len(ids_to_delete)

    def count(self) -> int:
        return self.store.count(namespace=self.namespace, kind=ACTIVATION_MODE_KIND)


__all__ = [
    "ACTIVATION_MODE_KIND",
    "CapturedActivationMode",
    "capture_activation_mode",
    "DynamicGraftSynthesizer",
]
