from __future__ import annotations

import pytest
import torch

from core.substrate.prediction_error import OrganError, PredictionErrorVector
from core.swm import SWMSource


def test_record_and_get_round_trip():
    v = PredictionErrorVector()
    entry = v.record(source=SWMSource.GLINER2, error=0.3)
    assert isinstance(entry, OrganError)
    assert entry.source is SWMSource.GLINER2
    assert entry.error == 0.3
    assert v.get(SWMSource.GLINER2).error == 0.3


def test_overwrite_keeps_latest():
    v = PredictionErrorVector()
    v.record(source=SWMSource.LLAMA, error=0.5)
    v.record(source=SWMSource.LLAMA, error=0.1)
    assert v.get(SWMSource.LLAMA).error == 0.1


def test_record_rejects_out_of_range_error():
    v = PredictionErrorVector()
    with pytest.raises(ValueError):
        v.record(source=SWMSource.GLICLASS, error=-0.01)
    with pytest.raises(ValueError):
        v.record(source=SWMSource.GLICLASS, error=1.5)


def test_missing_organ_raises():
    v = PredictionErrorVector()
    with pytest.raises(KeyError):
        v.get(SWMSource.WHISPER)


def test_as_tensor_default_uses_insertion_order():
    v = PredictionErrorVector()
    v.record(source=SWMSource.GLINER2, error=0.1)
    v.record(source=SWMSource.GLICLASS, error=0.4)
    v.record(source=SWMSource.LLAMA, error=0.2)
    t = v.as_tensor()
    assert torch.allclose(t, torch.tensor([0.1, 0.4, 0.2]))


def test_as_tensor_with_explicit_order():
    v = PredictionErrorVector()
    v.record(source=SWMSource.GLINER2, error=0.1)
    v.record(source=SWMSource.LLAMA, error=0.2)
    t = v.as_tensor(sources=[SWMSource.LLAMA, SWMSource.GLINER2])
    assert torch.allclose(t, torch.tensor([0.2, 0.1]))


def test_as_tensor_missing_source_raises():
    v = PredictionErrorVector()
    v.record(source=SWMSource.LLAMA, error=0.2)
    with pytest.raises(KeyError):
        v.as_tensor(sources=[SWMSource.LLAMA, SWMSource.WHISPER])


def test_joint_free_energy_sums_errors():
    v = PredictionErrorVector()
    v.record(source=SWMSource.GLINER2, error=0.3)
    v.record(source=SWMSource.GLICLASS, error=0.4)
    assert abs(v.joint_free_energy() - 0.7) < 1e-6


def test_reset_clears_state():
    v = PredictionErrorVector()
    v.record(source=SWMSource.GLINER2, error=0.3)
    v.reset()
    assert len(v) == 0
    with pytest.raises(KeyError):
        v.get(SWMSource.GLINER2)


def test_sources_preserves_insertion_order():
    v = PredictionErrorVector()
    v.record(source=SWMSource.LLAMA, error=0.1)
    v.record(source=SWMSource.GLINER2, error=0.2)
    v.record(source=SWMSource.GLICLASS, error=0.3)
    assert v.sources() == [SWMSource.LLAMA, SWMSource.GLINER2, SWMSource.GLICLASS]
