"""Tests for the swarm membrane: peer reliability + conformal tool gating.

The swarm receives untrusted broadcasts from arbitrary nodes on the LAN.
``PeerQuarantine`` is the only path between the network thread and the local
event bus, and these tests pin the contract that:

* every payload reaching the bus carries a peer reliability tag,
* native-tool source is rejected unless a local conformal validator vouches
  for it,
* hallucinating peers shrink toward zero influence on Dirichlet preferences
  without crossing zero.
"""

from __future__ import annotations

import pytest

from core.learning.preference_learning import DirichletPreference
from core.swarm.peer_reliability import PeerReliabilityRegistry
from core.swarm.quarantine import PeerQuarantine, PeerRejected


class TestPeerReliabilityRegistry:
    def test_uniform_prior_yields_neutral_reliability(self):
        reg = PeerReliabilityRegistry()
        assert reg.reliability_for("peer-A") == 0.5

    def test_low_error_observations_increase_reliability(self):
        reg = PeerReliabilityRegistry()
        for _ in range(10):
            reg.record_prediction_error("peer-A", 0.05)
        assert reg.reliability_for("peer-A") > 0.85

    def test_high_error_observations_decrease_reliability(self):
        reg = PeerReliabilityRegistry()
        for _ in range(10):
            reg.record_prediction_error("peer-B", 0.95)
        assert reg.reliability_for("peer-B") < 0.15

    def test_reliability_never_crosses_zero(self):
        reg = PeerReliabilityRegistry()
        for _ in range(1000):
            reg.record_prediction_error("peer-C", 1.0)
        rel = reg.reliability_for("peer-C")
        assert rel > 0.0
        assert rel < 0.01

    def test_invalid_error_raises(self):
        reg = PeerReliabilityRegistry()
        with pytest.raises(ValueError):
            reg.record_prediction_error("peer-A", -0.1)
        with pytest.raises(ValueError):
            reg.record_prediction_error("peer-A", 1.1)


class TestPeerQuarantineTagging:
    def test_ordinary_event_gets_peer_tag_and_reliability(self):
        reg = PeerReliabilityRegistry()
        q = PeerQuarantine(reliability=reg)
        tagged = q.intercept("chat.frame", {"intent": "hello"}, "peer-A")
        assert tagged["intent"] == "hello"
        assert tagged["_peer_id"] == "peer-A"
        assert 0.0 < tagged["_peer_reliability"] < 1.0

    def test_non_dict_payload_is_coerced(self):
        reg = PeerReliabilityRegistry()
        q = PeerQuarantine(reliability=reg)
        tagged = q.intercept("chat.frame", "hello world", "peer-A")
        assert tagged["_raw"] == "hello world"
        assert tagged["_peer_id"] == "peer-A"

    def test_control_plane_topics_skip_reliability_tag(self):
        reg = PeerReliabilityRegistry()
        q = PeerQuarantine(reliability=reg)
        tagged = q.intercept("swarm.peer.joined", {"node_id": "peer-A"}, "peer-A")
        assert tagged["_peer_id"] == "peer-A"
        assert "_peer_reliability" not in tagged


class TestPeerQuarantineToolValidation:
    def test_native_tool_source_without_validator_is_rejected(self):
        reg = PeerReliabilityRegistry()
        q = PeerQuarantine(reliability=reg, tool_validator=None)
        payload = {"source": "def f(x): return x", "sample_inputs": [{}], "domain": [0, 1]}
        with pytest.raises(PeerRejected):
            q.intercept("native_tool.synthesized", payload, "peer-A")
        assert q.stats["rejected"] == 1

    def test_native_tool_source_passes_when_validator_accepts(self):
        reg = PeerReliabilityRegistry()
        called: list = []

        def validator(payload):
            called.append(payload)

        q = PeerQuarantine(reliability=reg, tool_validator=validator)
        payload = {"source": "def f(x): return x", "sample_inputs": [{}], "domain": [0, 1]}
        tagged = q.intercept("native_tool.synthesized", payload, "peer-A")
        assert tagged["source"] == "def f(x): return x"
        assert "_peer_reliability" in tagged
        assert called

    def test_native_tool_rejection_propagates_when_validator_raises(self):
        reg = PeerReliabilityRegistry()

        def validator(payload):
            raise RuntimeError("conformal singleton not satisfied")

        q = PeerQuarantine(reliability=reg, tool_validator=validator)
        payload = {"source": "def f(x): return x", "sample_inputs": [{}], "domain": [0, 1]}
        with pytest.raises(PeerRejected):
            q.intercept("native_tool.synthesized", payload, "peer-B")

    def test_topic_with_native_tool_prefix_but_missing_fields_passes_through(self):
        reg = PeerReliabilityRegistry()
        q = PeerQuarantine(reliability=reg, tool_validator=None)
        # native_tool.drift carries metadata but no executable source — must not be rejected.
        tagged = q.intercept("native_tool.drift", {"tool": "humidity"}, "peer-A")
        assert tagged["tool"] == "humidity"
        assert "_peer_reliability" in tagged


class TestDirichletPeerSignal:
    def test_hallucinator_decays_toward_zero_without_crossing(self):
        reg = PeerReliabilityRegistry()
        q = PeerQuarantine(reliability=reg)
        for _ in range(200):
            reg.record_prediction_error("peer-bad", 0.99)

        pref = DirichletPreference(3, prior_strength=2.0)
        before = list(pref.alpha)
        bad_payload = q.intercept("chat.frame", {"obs": 2}, "peer-bad")

        for _ in range(50):
            pref.update_from_peer_signal(2, bad_payload, polarity=1.0, base_weight=1.0)

        # alpha[2] should have grown only marginally, alphas remain strictly positive
        assert pref.alpha[2] - before[2] < 5.0
        assert all(a > 0.0 for a in pref.alpha)

    def test_trusted_peer_shifts_mass_substantially(self):
        reg = PeerReliabilityRegistry()
        q = PeerQuarantine(reliability=reg)
        for _ in range(50):
            reg.record_prediction_error("peer-good", 0.02)

        pref = DirichletPreference(3, prior_strength=2.0)
        good_payload = q.intercept("chat.frame", {"obs": 0}, "peer-good")

        before_mass = pref.mean[0]
        for _ in range(50):
            pref.update_from_peer_signal(0, good_payload, polarity=1.0, base_weight=1.0)
        after_mass = pref.mean[0]
        assert after_mass > before_mass + 0.3

    def test_payload_without_reliability_tag_is_rejected(self):
        pref = DirichletPreference(3, prior_strength=2.0)
        with pytest.raises(ValueError):
            pref.update_from_peer_signal(0, {"obs": 1}, polarity=1.0, base_weight=1.0)
