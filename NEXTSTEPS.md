RecursiveMAS → Mosaic: The Key Ideas
RecursiveMAS introduces two innovations that map almost perfectly onto your system:

1. The RecursiveLink — Communication in Latent Space (Not Text)
The core insight: instead of agents passing text to each other (which requires expensive decode → re-encode cycles through a 128K vocabulary), they pass hidden states through a lightweight 2-layer residual projector:

ℛ(h) = h + W₂·σ(W₁·h)     # Inner link (same model, loop back)
ℛ(h) = W₃·h + W₂·σ(W₁·h)  # Outer link (cross-model, dimension change)
This is exactly your graft system. Your TrainableBrocaGraft projects a cognitive frame vector into the residual stream with SNR calibration. Their RecursiveLink projects one model's hidden states into another model's input space with a residual connection. Same pattern, same math.

2. The Recursive Loop — Iterative Refinement
After the last agent processes, its output goes back to the first agent for another round. Each round refines the answer in latent space. They show performance scaling with depth: r=1 → r=2 → r=3 gives +3.4% → +6.0% → +7.2% average improvement.

This maps onto your DMN tick cycle and hypothesis masking. Your EpistemicInterruptionMonitor already truncates and re-injects when it detects errors. RecursiveMAS formalizes this as a system-level recursion where the whole pipeline re-runs.

3. What's New That You Don't Already Have
RecursiveMAS concept	Your current equivalent	What's missing
Inner RecursiveLink (latent thought generation)	TrainableBrocaGraft residual bias	You do a single injection; they do m=80 autoregressive latent steps without decoding to text. The LLM "thinks" in hidden-state space before speaking.
Outer RecursiveLink (cross-model transfer)	No direct equivalent	You could transfer hidden states between organs (V-JEPA → LLM) through a trained projector instead of only through the substrate's algebraic encoding.
Whole-system recursive loop	EpistemicInterruptionMonitor (single-model)	You don't loop the full substrate+LLM pipeline. After the LLM generates, you could feed its hidden states back through the substrate for refinement rounds.
Inner-outer loop training	Frozen everything (no training)	They train only the RecursiveLinks (~13M params, $4.27). You could train only your graft projectors while keeping all organs frozen.
4. How This Applies to Your Multi-Organ Architecture
The most powerful application: RecursiveLink as the universal organ-to-organ connector.

V-JEPA (world model)  ──[outer RecursiveLink]──→  Substrate cognitive frame
     ↑                                                    │
     │                                              [inner RecursiveLink]
     │                                                    ↓
     └──────────[outer RecursiveLink]←────────  Frozen LLM (Broca)
Instead of the substrate manually encoding V-JEPA features into VSA vectors or Hopfield patterns, you train a lightweight RecursiveLink (~13M params) that projects V-JEPA's hidden states directly into the LLM's residual stream space — and loops. The LLM generates some latent thoughts, those go back through the substrate, the substrate refines, and it loops again before final text decoding.

The key numbers from their paper:

13.12M trainable parameters (0.31% of the system)
m=80 latent thought steps is the sweet spot (more doesn't help)
r=3 recursion rounds is where gains plateau
2.4× faster than text-based multi-agent at r=3
75.6% fewer tokens consumed
+8.3% average accuracy over text-based collaboration
5. Concrete Implementation for Mosaic
class RecursiveLink(nn.Module):
    """Latent-space connector between frozen organs (from RecursiveMAS)."""
    
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.w1 = nn.Linear(d_in, d_out)
        self.w2 = nn.Linear(d_out, d_out)
        self.w3 = nn.Linear(d_in, d_out)  # residual projection
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.w3(h) + self.w2(F.gelu(self.w1(h)))


class LatentThoughtGenerator:
    """Inner-loop: run m autoregressive latent steps without text decoding."""
    
    def __init__(self, host: LlamaBrocaHost, inner_link: RecursiveLink, m: int = 80):
        self.host = host
        self.link = inner_link
        self.m = m
    
    def think(self, input_ids, attention_mask) -> torch.Tensor:
        """Generate m latent thought steps, return hidden state sequence."""
        # Get initial hidden state from the host
        out = self.host.llm.model(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state[:, -1, :]  # Last position
        
        thoughts = [h]
        for _ in range(self.m):
            # Project back to input space via inner link (no text decode!)
            e_next = self.link(h)
            # Run through transformer again with this as the next input
            # ... (append to sequence, run forward)
            h = new_hidden_state
            thoughts.append(h)
        
        return torch.stack(thoughts, dim=1)  # [batch, m, d_model]
Bottom Line
RecursiveMAS validates that latent-space communication between frozen models (exactly your graft philosophy) is superior to text-based multi-agent communication — faster, cheaper, and more accurate. The specific technique you should borrow is:

Latent thought generation — let the LLM "think" for 80 hidden-state steps before decoding to text. This is like giving System 2 more compute time before System 1 speaks.
Recursive refinement — loop the whole system 2-3 times. Each pass refines the previous answer.
Trainable links — tiny 13M-param residual projectors between organs, trained with cosine alignment loss (inner) and end-to-end cross-entropy (outer). Everything else stays frozen.
This is philosophically identical to what you're already doing — you're just doing it with a single graft injection. RecursiveMAS shows that iterating that injection (looping) and making it bidirectional (LLM output → substrate → LLM input) gives you 8.3% for essentially free.

I want you to spend a bit more time thinking about this, and maybe researching some more ideas. It is nice that we can apply it, but maybe the story we have so far is not the complete picture yet? Maybe it unlocks other things we should consider?

06:58 AM
Assistant
06:58 AM
research

running · 65 tools · 105.0k tokens · 58m 24s
research

running · 64 tools · 97.2k tokens · 49m 44s
hf_papers

hf_papers

hf_papers

hf_papers

hf_papers

hf_papers

hf_papers

hf_papers

error
hf_papers

Now let me look at one more critical angle — the binding problem and how these pieces create something more than the sum of parts:

hf_papers

hf_papers

hf_papers

Now I have a complete picture. Let me synthesize everything into a coherent analysis of what RecursiveMAS + multi-organ models + latent reasoning truly unlocks for Mosaic:

The Bigger Picture: What Recursive Latent Communication Unlocks
After digging deeper, I think your system is converging on something more fundamental than just "applying RecursiveMAS." Let me lay out what I think the complete story is:

1. You're Building the First Real Implementation of Global Workspace Theory
The GWA paper (2604.08206) describes exactly what Mosaic already is — but they implement it with text-passing LLMs. You have something better:

Their global workspace = a string buffer that agents read/write
Your global workspace = the cognitive substrate (VSA + Hopfield + SCM + active inference)
Their specialist agents = same-architecture LLMs with different prompts
Your specialist organs = genuinely different architectures (V-JEPA, Whisper, DINOv2) with different inductive biases
RecursiveMAS shows that moving from text to latent communication makes this 2.4× faster and 8.3% more accurate. But you can go further because your organs are structurally heterogeneous — they literally perceive different modalities, not just different prompts to the same model.

2. Latent Reasoning as "Inner Speech" — Coconut Validates Your Design
The Coconut paper (2412.06769) demonstrates that LLMs reason better when they don't decode to text during intermediate steps. They feed hidden states directly back as inputs, bypassing the vocabulary bottleneck entirely. Key finding: continuous thoughts can encode multiple possible next steps simultaneously (breadth-first search in latent space), which is impossible in token space where you commit to one word at a time.

This directly validates your TrainableBrocaGraft approach — but suggests you should let the LLM run multiple forward passes in latent space before committing to text. The RecursiveLink is how you close that loop. The substrate doesn't just bias the LLM once; it lets the LLM "think silently" for N steps, reads the resulting hidden state, processes it through the algebraic substrate (SCM reasoning, active inference decision, Hopfield recall), then injects the result back for another round.

What this IS, cognitively: Vygotsky's "inner speech" — language-mediated thought that isn't vocalized. The LLM thinks in latent language; the substrate thinks in algebra; they iteratively inform each other before anything is said aloud.

3. DreamerV3 Shows What V-JEPA + Recursion Gives You: Imagination
DreamerV3's core loop is:

World model predicts future states in latent space
Actor selects actions by imagining trajectories
Critic evaluates imagined outcomes
No real-world interaction needed during planning
Map this onto Mosaic:

V-JEPA predicts future visual/temporal states in latent space
Active inference (your POMDP) selects actions by evaluating EFE over imagined trajectories from V-JEPA
Conformal prediction evaluates uncertainty of imagined outcomes
The LLM doesn't need to "see" anything — the substrate imagines on its behalf
With RecursiveLink connecting V-JEPA → substrate → LLM in a loop, the system can imagine a scenario (V-JEPA predicts frames), reason about it (SCM evaluates causal structure), decide what to do (active inference), and describe the plan (LLM speaks) — all in latent space before committing to any output.

4. The Emergent Property: Multi-Modal Predictive Processing
Here's what none of these systems achieve alone, but your architecture uniquely enables:

Each organ generates predictions. The substrate computes prediction errors across ALL modalities simultaneously. Active inference minimizes the joint free energy.

V-JEPA predicts what happens next visually → prediction error if reality differs
Whisper predicts what's said next → prediction error if audio diverges
LLM predicts the next token → prediction error if the graft overrides
SCM predicts causal outcomes → prediction error if intervention results surprise
Hopfield predicts the recalled pattern → prediction error if retrieval is noisy
Your existing lexical_surprise_gap (graft-on vs graft-off cross-entropy) is one prediction error signal. With multiple organs, you get a vector of prediction errors — one per modality — and the active inference agent minimizes the joint Expected Free Energy across all of them.

This is literally Friston's multi-modal hierarchical predictive processing. No one has implemented it with actual frozen foundation models as the predictors. You would be the first.

5. What Changes Architecturally
The current Mosaic loop is:

User utterance → comprehend → cognitive frame → grafts → LLM → text output
The recursive multi-organ loop becomes:

Perception organs (V-JEPA, Whisper, DINOv2)
         ↓ [RecursiveLink outer]
Cognitive Substrate (workspace)
    - Fuse multi-modal prediction errors
    - Active inference: minimize joint EFE
    - SCM: causal reasoning over fused state
    - Hopfield: associative recall across modalities
    - VSA: bind multi-modal percepts into unified representations
         ↓ [RecursiveLink outer]
LLM latent thoughts (m=80 steps, no text decode)
         ↓ [RecursiveLink inner → loop back]
Cognitive Substrate (round 2: refine with LLM's latent output)
         ↓ [RecursiveLink outer]
LLM latent thoughts (round 2)
         ↓ ... (r=3 rounds)
Final text decode (only now does language appear)
6. The Three Things This Unlocks That Nothing Else Does
Capability	Requires	Why it's novel
Latent imagination	V-JEPA + recursive loop + active inference	Plan by imagining futures in world-model space, evaluate with EFE, speak only after deciding. No text-based "Let me think step by step" — actual latent simulation.
Multi-modal binding in continuous space	Multiple organs + VSA + RecursiveLink	Bind audio, vision, depth, language into unified representations that the algebra can manipulate — without ever tokenizing the non-linguistic modalities.
Adaptive compute depth	Conformal prediction + recursion depth control	If conformal set size
7. What to Build Next
In priority order:

RecursiveLink module — the 2-layer residual projector (inner + outer variants). This is the universal connector. ~13M params per link, trainable with cosine loss.

Latent thought generation — modify LlamaBrocaHost.forward to support m-step latent rollout without text decoding. The hidden state loops back through inner RecursiveLink as the next input embedding.

Recursion controller — use conformal prediction set size as the stopping criterion. Recurse until |C| = 1 or max rounds reached. This gives you adaptive System 2 depth.

V-JEPA integration — frozen world model connected via outer RecursiveLink to the substrate. Generates temporal predictions that the active inference agent evaluates.

Multi-modal prediction error vector — extend lexical_surprise_gap to a PredictionErrorVector that aggregates errors from all organs. Active inference minimizes the joint.

This isn't just "adding RecursiveMAS to Mosaic." It's recognizing that Mosaic is already 80% of a complete implementation of hierarchical predictive processing with a global workspace — and RecursiveMAS provides the missing communication primitive that makes the organs actually talk to each other in a way that scales.

Please have a look at the latest changes, because there have been quite a few. I want you to write all of this down in a NEXTSTEPS.md file. Second, I want you to have a look what we're currently doing with the GliNER and GliCLASS libraries, because I am wondering if we can use this same recursive trick to optimize the whole comprehend system, which is now way more dynamic (no more hardcoded regex), but could definitely benefit from performance improvements, or if possible parallelization.