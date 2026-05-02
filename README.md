# MOSAIC

> **The LLM is locked in a glass box. It only speaks. Everything else —
> memory, reasoning, perception, planning, emotion, causal inference —
> lives in a persistent cognitive substrate that slips intelligent notes
> through the vents of the residual stream.**

---

## The thesis

Today's LLMs are extraordinary surface-form generators trapped inside an
architecture that asks them to *also* be world models, planners, memory
stores, and causal reasoners. They are none of those things. They are
associative language cortex — and that is all this system asks of them.

MOSAIC demotes the LLM to a **speech interface**: a frozen decoder whose
weights are never updated, whose only job is to produce fluent language.
All higher cognition is handled by a **cognitive substrate** built from
components with published mathematical guarantees. The substrate
communicates with the LLM exclusively through **grafts** — small modules
that bias the residual stream and logit distribution at every decoding
step, without consuming prompt tokens and without touching frozen weights.

This is not an engineering shortcut to save on training cost. It is a
**deliberate architectural choice** that prevents catastrophic forgetting.
When you fine-tune a model to learn a new fact, you degrade its existing
knowledge. When you inject knowledge through a graft, the base model's
capabilities remain bit-for-bit identical. The substrate can learn
continuously — accumulate memories, revise beliefs, discover causal
structure, compile habits — while the language organ stays pristine.

```
┌─────────────────────────────────────────────────────────┐
│  Cognitive Substrate (System 2 — learns continuously)   │
│                                                         │
│  perception · memory · reasoning · planning · emotion   │
│                                                         │
│         ┌──────────────────────────────┐                │
│         │    Trainable Grafts          │                │
│         │  (residual bias, logit bias, │                │
│         │   lexical plan — per step)   │                │
│         └──────────────┬───────────────┘                │
└────────────────────────┼────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Frozen LLM (System 1 — never changes)                  │
│                                                         │
│  "The glass box" — produces fluent language, nothing    │
│  else. Weights locked. Vocabulary locked. The graft     │
│  forces it to describe ideas it has no words for by     │
│  inventing metaphors from its existing subword space.   │
└─────────────────────────────────────────────────────────┘
```

---

## The cognitive organ matrix

Every component has a **job title**. The matrix defines what each organ
does, what it outputs, and where that output flows next.

### Perceptual organs (frozen pre-trained models)

| Organ | Brain analogy | Model | Output | Flows to |
|-------|---------------|-------|--------|----------|
| **Language** (Broca's) | Broca's area | Llama 3.2 1B | Token stream | User |
| **Visual cortex** | V1–V4 | DINOv2-Large (307M) | `[1024]` feature vector | Substrate frames |
| **Ventral stream** | Inferotemporal | I-JEPA ViT-H (632M) | `[1280]` semantic features | Substrate frames |
| **Dorsal stream** | Area MT/MST | V-JEPA2 ViT-H (632M) | `[1280]` temporal prediction | World model / SCM |
| **Spatial cortex** | Parietal | Depth Anything V2 (335M) | `[1024]` depth + spatial stats | Substrate frames |
| **Auditory cortex** | A1 + association | Whisper-turbo (809M) | Transcription + `[1280]` audio embedding | Extraction organ → Memory |
| **Association cortex** | Superior temporal sulcus | ImageBind (1.13B) | `[1024]` shared multi-modal embedding | Cross-modal Hopfield retrieval |

### Language understanding organs (frozen encoders, <10ms per utterance)

| Organ | Brain analogy | Model | Output | Flows to |
|-------|---------------|-------|--------|----------|
| **Extraction** (Wernicke's) | Wernicke's area | GLiNER2 (205M) | Entities + relations + intent labels | Semantic memory, SCM, Router |
| **Affect** | Limbic / insula | GoEmotions (125M) | 28 emotions + valence + arousal | Preference learning, Hawkes, Active inference |

### Algebraic substrate (pure math, no learned weights)

| Organ | Brain analogy | Job title | Input → Output |
|-------|---------------|-----------|----------------|
| **VSA / HRR** | Hippocampal binding | Zero-shot analogy via circular convolution | Concepts → `[10000]` bound hypervector |
| **Hopfield** | Hippocampal retrieval | Content-addressable pattern completion | Noisy query → nearest stored pattern |
| **Hawkes** | Working memory heat | Temporal intuition — conversational pacing | Event stream → decay-weighted intensity `float` |
| **Conformal** | Uncertainty estimation | Coverage-guaranteed set prediction | Softmax dist → prediction set with `P[y∈C] ≥ 1−α` |
| **SCM** | Causal reasoning | `do(·)` calculus, counterfactuals, backdoor adjustment | Intervention query → probability `float` |
| **Active inference** | Decision-making | EFE minimization over POMDPs | Belief state → action + posterior entropy |
| **Dirichlet preference** | Personality / values | Bayesian preference learning | Feedback signal → updated `C` vector |

### Control pathways (grafts — the corticocortical connections)

| Graft | Where it injects | What it does |
|-------|-----------------|--------------|
| **TrainableFeatureGraft** | `final_hidden` | Projects cognitive frame into residual stream at calibrated SNR |
| **LexicalPlanGraft** | `final_hidden` | Biases toward a speech plan of substrate-chosen tokens |
| **LogitBiasGraft** | `logits` | Content-aware subword bonus from frame subject/predicate/answer |
| **HypothesisMaskingGraft** | `logits` | Physically blocks rejected tokens via negative logit bias |
| **CausalConstraintGraft** | KV memory | Pulls LLM toward SCM's `P(Y|do(T=t))` when attending cause concept |
| **ModalityShiftGraft** | `final_hidden` | Injects cognitive mood direction (analytical, fluent, etc.) |

### Infrastructure

| System | Job title | Data flow |
|--------|-----------|-----------|
| **EventBus** | Global workspace / blackboard | All organs publish; all organs subscribe |
| **Swarm** | Inter-node UDP multicast | Every EventBus event flows to LAN peers and back |
| **Knowledge crawler** | Web perception | URLs → Trafilatura → GLiNER2 → Semantic memory |
| **DMN** | Background processing | Consolidation, separation, discovery, chunking, tool foraging, REM |
| **Self-improve** | Meta-learning daemon | Propose patch → Docker validate → PR → re-benchmark |

---

## The lifecycle of a thought

To understand how these organs cooperate, trace a single piece of
knowledge from first encounter through to compiled reflex.

### Phase 1: Perception (System 2 — deliberate, slow)

The user says: *"Ada lives in Rome."*

1. The **Extraction organ** (GLiNER2) fires in <10ms:
   - Entities: `[("Ada", person, 0.94), ("Rome", location, 0.97)]`
   - Relation: `(ada, lives_in, rome, 0.91)`

2. The **Affect organ** (GoEmotions) fires in <5ms:
   - Dominant: `neutral` (0.72)
   - No cognitive state signals above threshold

3. The substrate's **CognitiveRouter** receives both outputs and constructs a
   `CognitiveFrame(intent="memory_write", subject="ada", answer="rome")`.

4. **Semantic memory** stores the triple with confidence 0.91 and provenance
   from the extraction organ. The **VSA codebook** binds
   `subject ⊗ ada + predicate ⊗ lives_in + object ⊗ rome` into a single
   10,000-dim hypervector and stores it in the **Hopfield memory**.

5. The **Hawkes process** records a `memory_write` event — the intensity on
   that channel spikes and begins exponential decay.

### Phase 2: Retrieval (System 2 — but getting faster)

Later, the user asks: *"Where is Ada?"*

1. **Extraction organ**: `intent=question`, entities: `[("Ada", person)]`
2. **Router**: `CognitiveFrame(intent="memory_lookup", subject="ada")`
3. **Semantic memory**: recalls `(ada, lives_in, rome, confidence=0.91)`
4. **Conformal predictor**: `|C| = 1` (only "rome" in the prediction set) → high confidence
5. **Grafts activate**: The `TrainableFeatureGraft` injects the frame features into the
   residual stream. The `LexicalPlanGraft` biases toward tokens `["ada", "is", "in", "rome"]`.
   The `LogitBiasGraft` boosts subwords of "rome".
6. **Frozen LLM** generates: *"Ada is in Rome."* — the grafts won, the LLM spoke what
   the substrate knew, without ever seeing "Ada" or "Rome" in its prompt.

### Phase 3: Consolidation (DMN — background, between turns)

While the user is silent, the Default Mode Network ticks:

1. **Consolidation**: Episode graph PageRank boosts confidence of central facts.
2. **Separation**: If another entity also has `lives_in = rome`, the DMN detects
   ambiguity (binary entropy) and prepares a clarifying-question cue.
3. **Latent discovery**: Random `do(·)` interventions on the SCM check if `ada.lives_in`
   causally affects any other variable.

### Phase 4: Compilation (System 2 → System 1)

After the pattern `memory_write → memory_lookup → answer` repeats many times:

1. The **DMNChunkingCompiler** detects the repeated intent sequence.
2. It averages the feature vectors of every instance into a single **compiled macro**.
3. On the next occurrence, the substrate skips the multi-step routing and injects
   the macro's feature vector directly — the thought has become a reflex.

This is the musician who no longer looks at the fretboard. The causal discovery
that was once slow, deliberate, conscious reasoning has been compiled into a
fast, automatic System 1 response. The documentation of this progression — from
first encounter through deliberate analysis to automatic execution — is the
architecture's most compelling feature.

### Phase 5: Ontological expansion (when the LLM has no words)

When the substrate discovers a novel concept via the PC algorithm — a causal
node that has no English name — the **Hebbian orthogonalization** module
(Gram-Schmidt) creates a new, mathematically independent axis in concept space.
The frozen LLM has no token for this concept. But the `TrainableFeatureGraft`
maps the new orthogonal vector into the closest available approximation within
the LLM's residual stream, forcing it to *invent a metaphor* — to describe the
negative space of the new idea using the subwords it already possesses.

---

## The swarm

Multiple MOSAIC instances on a LAN communicate freely via UDP multicast
(`239.255.77.1:50077`, TTL=1). There is no orchestration. Every EventBus
event flows to the network. Every network event flows to the local bus.
Peers discover each other automatically via heartbeat (2s interval, 8s
timeout). The substrate decides what to do with what it hears.

```
Node A (LLM + visual organs)  ←──UDP multicast──→  Node B (extraction + affect + memory)
         ↕                                                    ↕
Node C (causal SCM + active inference)  ←────────→  Node D (knowledge crawler)
```

---

## Quick start

```bash
make install                       # uv sync with all extras
export HF_TOKEN=hf_…              # for gated Llama checkpoint
make tui                           # full TUI with substrate
make chat                          # plain streaming CLI
make bench                         # benchmarks (standard + architecture probes)
make paper                         # regenerate LaTeX paper from benchmarks
```

---

## Project structure

```
core/
├── organs/              # Frozen specialist models (perception, affect, extraction)
├── cognition/           # Substrate controller, top-down control, predictive coding
├── causal/              # FiniteSCM, PC discovery, DAG utilities
├── memory/              # Hopfield, SQLite activation memory
├── symbolic/            # VSA / HRR algebra
├── temporal/            # Hawkes processes
├── calibration/         # Conformal prediction
├── grafting/            # All graft types + dynamic graft synthesis
├── host/                # Frozen LLM wrapper, tokenizer compatibility
├── agent/               # Active inference, POMDPs, coupled EFE
├── learning/            # Motor learning, preference learning
├── idletime/            # DMN: chunking, ontological expansion, repository
├── knowledge/           # Scrapy + Trafilatura web crawling pipeline
├── swarm/               # UDP multicast peer communication
├── frame/               # Continuous cognitive frame encoding
├── substrate/           # Runtime config, episode graph
├── workers/             # Self-improvement Docker daemon
├── natives/             # Native tool synthesis + sandbox
├── benchmarks/          # HF datasets, lm-eval, substrate-specific benchmarks
├── paper/               # Benchmark-to-LaTeX harness
├── tui/                 # Textual chat + benchmark dashboards
├── chat/                # CLI REPL
├── system/              # Device, event bus, control plane
└── experiments/         # Demo runners
```

---

## Tests

```bash
pytest -q
```

Tests exercise the algebra (VSA capacity, Hopfield retrieval, Hawkes
excitation, conformal coverage), the belief revision engine (poison
resistance), top-down control (masking converges, interruption fires,
modality shifts direction, causal constraints pull toward SCM verdict),
the DMN lifecycle (REM only fires when idle), and the graft slot system
(hooks install/remove correctly, SNR scaling matches target). No model
downloads.

---

## Glossary

| Term | Definition |
|------|-----------|
| **Graft** | A module spliced into the frozen LLM's forward pass. The substrate's only channel into the language organ. |
| **Cognitive frame** | A non-linguistic content packet (`intent`, `subject`, `answer`, `confidence`, `evidence`) that the grafts translate into residual-stream and logit biases. |
| **SCM** | Structural Causal Model. DAG + structural equations. Supports `do(·)` interventions, counterfactuals, backdoor/frontdoor adjustment. |
| **EFE** | Expected Free Energy. The quantity active inference minimizes — balancing pragmatic value (reach preferred observations) with epistemic value (reduce uncertainty). |
| **VSA/HRR** | Vector Symbolic Architecture / Holographic Reduced Representations. Bind and unbind concepts via circular convolution in O(d log d). |
| **Hopfield** | Modern Continuous Hopfield Network. One-step content-addressable retrieval with exponential storage capacity in the embedding dimension. |
| **Hawkes** | Multivariate self-exciting point process. Each event raises the intensity of future events on the same and related channels, with exponential decay. The substrate's sense of conversational "heat." |
| **Conformal** | Split-conformal prediction. Turns any scoring model into a set predictor with marginal coverage guarantee `P[y ∈ C(x)] ≥ 1−α`. Set size > 1 = Fristonian ambiguity signal. |
| **DMN** | Default Mode Network. Background daemon that runs consolidation, separation, latent discovery, chunk compilation, and tool foraging between user turns. |
| **Swarm** | UDP multicast peer communication. All events flow freely between LAN nodes. No orchestration. |
| **Organ** | A frozen pre-trained model serving a specific cognitive function. The LLM is one organ (language production). DINOv2, Whisper, GLiNER2, GoEmotions are others. |
