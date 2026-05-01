"""Tests for the tool-foraging POMDP and ``synthesize_tool`` action.

These tests confirm that the Expected Free Energy mathematics actually
forces the substrate to choose ``synthesize_tool`` precisely when the
substrate is confused (high prior on ``knowledge_insufficient``) and few
tools exist, and to prefer ``use_existing_tool`` when the toolbox is
mature and the substrate is confident.
"""

from __future__ import annotations

import pytest

from asi_broca_core.active_inference import (
    ActiveInferenceAgent,
    CategoricalPOMDP,
    ToolForagingAgent,
    build_tool_foraging_pomdp,
    extend_pomdp_with_synthesize_tool,
    build_tiger_pomdp,
)


# ---------------------------------------------------------------------------
# build_tool_foraging_pomdp
# ---------------------------------------------------------------------------


def test_pomdp_has_synthesize_tool_action_and_two_states():
    p = build_tool_foraging_pomdp(n_existing_tools=0, insufficient_prior=0.5)
    assert "synthesize_tool" in p.action_names
    assert p.state_names == ["knowledge_sufficient", "knowledge_insufficient"]
    assert p.observation_names == ["info_gained", "info_stagnant"]
    # Three discrete actions; identity transitions over the two latent states.
    assert p.n_actions == 3
    assert p.n_states == 2


def test_likelihoods_increase_existing_tool_coverage_with_more_tools():
    p_no_tools = build_tool_foraging_pomdp(n_existing_tools=0)
    p_with_tools = build_tool_foraging_pomdp(n_existing_tools=10)
    use_idx = p_no_tools.action_names.index("use_existing_tool")
    suff_idx = p_no_tools.state_names.index("knowledge_sufficient")
    # P(info_gained | sufficient, use_existing_tool) must be strictly higher with more tools.
    assert p_with_tools.A[use_idx][0][suff_idx] > p_no_tools.A[use_idx][0][suff_idx]


def test_efe_picks_synthesize_when_confused_and_no_tools():
    p = build_tool_foraging_pomdp(n_existing_tools=0, insufficient_prior=0.9)
    agent = ActiveInferenceAgent(p, horizon=1, learn=False)
    decision = agent.decide()
    assert decision.action_name == "synthesize_tool"


def test_efe_picks_use_existing_when_confident_and_many_tools():
    p = build_tool_foraging_pomdp(n_existing_tools=10, insufficient_prior=0.05)
    agent = ActiveInferenceAgent(p, horizon=1, learn=False)
    decision = agent.decide()
    assert decision.action_name == "use_existing_tool"


def test_synthesize_efe_decreases_when_confusion_grows():
    """Holding tool count fixed, increasing the prior on insufficiency must
    *not increase* synthesize_tool's expected free energy — it should make it
    more attractive (lower G)."""

    base = build_tool_foraging_pomdp(n_existing_tools=2, insufficient_prior=0.1)
    confused = build_tool_foraging_pomdp(n_existing_tools=2, insufficient_prior=0.9)

    def _g(p, action_name):
        a_idx = p.action_names.index(action_name)
        return p.evaluate_policy([a_idx]).expected_free_energy

    g_base = _g(base, "synthesize_tool")
    g_conf = _g(confused, "synthesize_tool")
    assert g_conf < g_base


# ---------------------------------------------------------------------------
# ToolForagingAgent
# ---------------------------------------------------------------------------


def test_tool_foraging_agent_should_synthesize_high_confusion_zero_tools():
    agent = ToolForagingAgent.build(n_existing_tools=0, insufficient_prior=0.95)
    assert agent.should_synthesize() is True


def test_tool_foraging_agent_should_not_synthesize_when_satisfied():
    agent = ToolForagingAgent.build(n_existing_tools=8, insufficient_prior=0.05)
    assert agent.should_synthesize() is False


def test_tool_foraging_agent_update_belief_changes_decision():
    agent = ToolForagingAgent.build(n_existing_tools=0, insufficient_prior=0.05)
    # With low confusion + zero tools, exploration should beat synthesis (or at
    # least synthesis should not be chosen).
    initial = agent.decide()
    assert initial.action_name != "synthesize_tool"
    # Now confuse the agent — synthesize should become the EFE-minimising action.
    agent.update_belief(insufficient_prior=0.95)
    after = agent.decide()
    assert after.action_name == "synthesize_tool"


def test_tool_foraging_agent_observe_updates_belief():
    """observe() should update the Bayesian belief over latent states; if we
    keep gaining info from synthesize_tool, the posterior must collapse onto
    'knowledge_insufficient' (the state where synthesize_tool produces info)."""

    agent = ToolForagingAgent.build(n_existing_tools=0, insufficient_prior=0.5)
    # Repeatedly observe info_gained after synthesize_tool: posterior over latent state
    # must shift toward knowledge_insufficient (where synthesizing yields info).
    for _ in range(3):
        agent.observe("synthesize_tool", "info_gained")
    insuff_idx = agent.pomdp.state_names.index("knowledge_insufficient")
    suff_idx = agent.pomdp.state_names.index("knowledge_sufficient")
    assert agent.agent.qs[insuff_idx] > agent.agent.qs[suff_idx]


# ---------------------------------------------------------------------------
# extend_pomdp_with_synthesize_tool
# ---------------------------------------------------------------------------


def test_extend_pomdp_preserves_original_actions_and_states():
    base = build_tiger_pomdp()
    extended = extend_pomdp_with_synthesize_tool(base, n_existing_tools=2)
    assert extended.action_names == list(base.action_names) + ["synthesize_tool"]
    assert extended.state_names == base.state_names
    assert extended.observation_names == base.observation_names
    # Original action likelihoods preserved.
    for a in range(base.n_actions):
        for o in range(base.n_observations):
            for s in range(base.n_states):
                assert pytest.approx(extended.A[a][o][s]) == base.A[a][o][s]


def test_extend_pomdp_does_not_mutate_input():
    base = build_tiger_pomdp()
    base_actions = list(base.action_names)
    extend_pomdp_with_synthesize_tool(base, n_existing_tools=1)
    assert base.action_names == base_actions  # untouched


def test_extended_pomdp_synth_transition_is_identity():
    base = build_tiger_pomdp()
    extended = extend_pomdp_with_synthesize_tool(base, n_existing_tools=1)
    new_action_idx = extended.action_names.index("synthesize_tool")
    for s in range(extended.n_states):
        for sp in range(extended.n_states):
            expected = 1.0 if sp == s else 0.0
            assert pytest.approx(extended.B[new_action_idx][sp][s], abs=1e-6) == expected


def test_extended_pomdp_synth_likelihood_is_well_formed():
    base = build_tiger_pomdp()
    extended = extend_pomdp_with_synthesize_tool(base, n_existing_tools=4)
    new_action_idx = extended.action_names.index("synthesize_tool")
    # For each latent state, the observation distribution must sum to 1.
    for s in range(extended.n_states):
        col = [extended.A[new_action_idx][o][s] for o in range(extended.n_observations)]
        assert pytest.approx(sum(col), abs=1e-6) == 1.0
