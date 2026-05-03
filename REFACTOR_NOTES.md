# Refactor notes

## What changed

- Split the active-inference monolith into small composable classes:
  - `DistributionMath`
  - `CategoricalPOMDP`
  - `ActiveInferenceAgent`
  - `CoupledEFEAgent`
  - `POMDPBuilder`
  - `ToolForagingPOMDPBuilder`
  - `TigerDoorEnv`
  - `TigerEpisodeRunner`
- Replaced `core.agent.active_inference` with a compatibility facade whose public callables are bound methods on `ActiveInferenceFacade`.
- Reworked `core.__init__` and `core.encoders.__init__` to use lazy exports so importing one concern no longer eagerly imports the whole substrate, host, memory, and encoder stack.
- Refactored `core.cli` into composable runtime classes for session setup, model preparation, controller construction, background services, logging, and workspace creation. Historical CLI callables are now bound facade methods.
- Decoupled `SubstrateController` from heavy top-level imports. Comprehension, macro, and host-specific imports are now loaded at the methods that need them.
- Moved background-worker and self-improvement lifecycle state into `SubstrateSessionState`, with a compatibility property for old `mind._self_improve_worker` access.
- Replaced the activation-memory SQLModel/SQLAlchemy implementation with raw SQLite classes and a tiny repository layer:
  - `SQLiteActivationMemory`
  - `SQLiteActivationConnection`
  - `SQLiteActivationSchema`
  - `SQLiteActivationContext`
  - plain row models in `core.memory.model`
- Removed the legacy monolith backup files and removed all `sqlalchemy` / `sqlmodel` imports from `core`.

## Validation run in this environment

The following chunks passed:

- `tests/test_affect_trace.py` through `tests/test_docker_self_improve_worker.py`: 85 passed, 1 skipped.
- Targeted post-CLI-refactor smoke: `tests/test_affect_trace.py`, `tests/test_broca_snapshot.py`, `tests/test_docker_self_improve_worker.py`, `tests/test_memory_layers.py`, `tests/test_multimodal_perception_wiring.py`, and `tests/test_relation_extraction_and_consolidation.py`: 37 passed, 1 skipped.
- Targeted post-CLI-refactor smoke: `tests/test_substrate_intent_gating.py`, `tests/test_substrate_memory_fidelity.py`, `tests/test_tool_foraging.py`, and `tests/test_top_down_control.py`: 68 passed.
- `tests/test_dynamic_grafts.py` through `tests/test_motor_learning.py`: 103 passed, 1 skipped.
- `tests/test_multimodal_perception_wiring.py` through `tests/test_relation_extraction_and_consolidation.py`: 56 passed.
- `tests/test_rem_sleep.py`, `tests/test_semantic_cascade.py`, `tests/test_substrate_intent_gating.py`, `tests/test_substrate_memory_fidelity.py`, and `tests/test_tool_foraging.py`: 33 passed.
- `tests/test_tui_chat_component_imports.py`, `tests/test_vision.py`, and `tests/test_vsa.py`: 16 passed.
- `tests/test_top_down_control.py`: 45 passed.
- `python -m compileall -q core research_lab`: passed.

Known environment-blocked tests:

- `tests/test_encoder_integration.py`: 4 passed, 14 errors because this environment does not have `gliner2` or `transformers` installed.
- `tests/test_substrate_persistence_contract.py`: 3 failures because the canonical controller path loads the real Llama host, which requires `transformers` in this environment.

No test failure observed above was caused by the new raw-SQLite activation-memory implementation or the active-inference split.
