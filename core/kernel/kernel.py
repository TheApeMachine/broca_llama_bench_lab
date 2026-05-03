"""Mosaic kernel: single runtime entrypoint for chat and future benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from ..frame import CognitiveFrame
from .builder import KernelBuilder
from .health import SystemHealth
from .manifest import RuntimeManifest


@dataclass
class AssistantTurn:
    """Structured output from one kernel chat call."""

    frame: CognitiveFrame
    text: str
    health: SystemHealth


class MosaicKernel:
    """Small façade over the currently constructed runtime."""

    def __init__(self, *, controller: Any, manifest: RuntimeManifest, health: SystemHealth) -> None:
        self.controller = controller
        self.manifest = manifest
        self._health = health

    @classmethod
    def from_manifest(
        cls,
        profile: str | None = None,
        *,
        manifest: RuntimeManifest | None = None,
        **controller_kwargs: Any,
    ) -> "MosaicKernel":
        result = KernelBuilder().build(profile=profile, manifest=manifest, **controller_kwargs)
        return cls(controller=result.controller, manifest=result.manifest, health=result.health)

    def health(self) -> SystemHealth:
        self._health = SystemHealth.from_controller(self.controller, manifest=self.manifest)
        return self._health

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_new_tokens: int = 256,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        on_token: Callable[[str], None] | None = None,
    ) -> AssistantTurn:
        frame, text = self.controller.chat_reply(
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            on_token=on_token,
        )
        return AssistantTurn(frame=frame, text=text, health=self.health())
