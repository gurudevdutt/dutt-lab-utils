"""
Tests for ClassifierRouter: first-backend accepted, confidence cascade, unavailable skip,
error fallthrough, all-fail raises, zero-threshold short-circuit, generate fallthrough.
Uses MagicMock backends; no HTTP.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from pittqlab_utils.llm import ClassifierRouter, ClassifyResult, GenerateResult


def _make_mock_backend(name: str, available: bool = True):
    b = MagicMock()
    b.name = name
    b.is_available = available
    b.ping = AsyncMock(return_value=None)
    return b


# ---------------------------------------------------------------------------
# First backend accepted
# ---------------------------------------------------------------------------

async def test_router_classify_first_backend_accepted():
    high = _make_mock_backend("high")
    high.classify = AsyncMock(return_value=ClassifyResult("A", 0.9, {}, "high"))
    low = _make_mock_backend("low")
    low.classify = AsyncMock(return_value=ClassifyResult("B", 0.5, {}, "low"))
    router = ClassifierRouter(backends=[high, low], confidence_threshold=0.75)
    high.is_available = True
    low.is_available = True
    result = await router.classify("hi", ["A", "B"])
    assert result.label == "A"
    assert result.confidence == 0.9
    high.classify.assert_called_once()
    low.classify.assert_not_called()


# ---------------------------------------------------------------------------
# Confidence cascade
# ---------------------------------------------------------------------------

async def test_router_classify_confidence_cascade():
    first = _make_mock_backend("first")
    first.classify = AsyncMock(return_value=ClassifyResult("A", 0.5, {}, "first"))
    second = _make_mock_backend("second")
    second.classify = AsyncMock(return_value=ClassifyResult("B", 0.8, {}, "second"))
    router = ClassifierRouter(backends=[first, second], confidence_threshold=0.75)
    first.is_available = True
    second.is_available = True
    result = await router.classify("hi", ["A", "B"])
    assert result.label == "B"
    assert result.confidence == 0.8
    first.classify.assert_called_once()
    second.classify.assert_called_once()


# ---------------------------------------------------------------------------
# Unavailable skip
# ---------------------------------------------------------------------------

async def test_router_classify_unavailable_skip():
    unav = _make_mock_backend("unav", available=False)
    unav.is_available = False
    unav.classify = AsyncMock(return_value=ClassifyResult("A", 0.9, {}, "unav"))
    avail = _make_mock_backend("avail")
    avail.classify = AsyncMock(return_value=ClassifyResult("B", 0.9, {}, "avail"))
    router = ClassifierRouter(backends=[unav, avail], confidence_threshold=0.75)
    result = await router.classify("hi", ["A", "B"])
    assert result.label == "B"
    unav.classify.assert_not_called()
    avail.classify.assert_called_once()


# ---------------------------------------------------------------------------
# Error fallthrough
# ---------------------------------------------------------------------------

async def test_router_classify_error_fallthrough():
    fail = _make_mock_backend("fail")
    fail.classify = AsyncMock(side_effect=RuntimeError("down"))
    ok = _make_mock_backend("ok")
    ok.classify = AsyncMock(return_value=ClassifyResult("A", 0.8, {}, "ok"))
    router = ClassifierRouter(backends=[fail, ok], confidence_threshold=0.75)
    result = await router.classify("hi", ["A"])
    assert result.label == "A"
    assert result.confidence == 0.8


# ---------------------------------------------------------------------------
# All fail raises
# ---------------------------------------------------------------------------

async def test_router_classify_all_fail_raises():
    a = _make_mock_backend("a")
    a.classify = AsyncMock(side_effect=RuntimeError("err"))
    b = _make_mock_backend("b", available=False)
    b.classify = AsyncMock()
    router = ClassifierRouter(backends=[a, b], confidence_threshold=0.75)
    with pytest.raises(RuntimeError, match="All backends failed or unavailable"):
        await router.classify("hi", ["X"])


async def test_router_classify_all_unavailable_raises():
    a = _make_mock_backend("a", available=False)
    a.classify = AsyncMock()
    router = ClassifierRouter(backends=[a], confidence_threshold=0.75)
    with pytest.raises(RuntimeError, match="All backends failed or unavailable"):
        await router.classify("hi", ["X"])


# ---------------------------------------------------------------------------
# Zero-threshold short-circuit
# ---------------------------------------------------------------------------

async def test_router_classify_zero_threshold_accepts_first():
    low_conf = _make_mock_backend("low")
    low_conf.classify = AsyncMock(return_value=ClassifyResult("A", 0.1, {}, "low"))
    high_conf = _make_mock_backend("high")
    high_conf.classify = AsyncMock(return_value=ClassifyResult("B", 0.9, {}, "high"))
    router = ClassifierRouter(backends=[low_conf, high_conf], confidence_threshold=0.0)
    result = await router.classify("hi", ["A", "B"])
    assert result.label == "A"
    assert result.confidence == 0.1
    low_conf.classify.assert_called_once()
    high_conf.classify.assert_not_called()


# ---------------------------------------------------------------------------
# Best result when none exceed threshold
# ---------------------------------------------------------------------------

async def test_router_classify_returns_best_when_none_exceed_threshold():
    first = _make_mock_backend("first")
    first.classify = AsyncMock(return_value=ClassifyResult("A", 0.5, {}, "first"))
    second = _make_mock_backend("second")
    second.classify = AsyncMock(return_value=ClassifyResult("B", 0.6, {}, "second"))
    router = ClassifierRouter(backends=[first, second], confidence_threshold=0.9)
    result = await router.classify("hi", ["A", "B"])
    assert result.label == "B"
    assert result.confidence == 0.6


# ---------------------------------------------------------------------------
# Generate fallthrough
# ---------------------------------------------------------------------------

async def test_router_generate_first_available_wins():
    unav = _make_mock_backend("unav", available=False)
    unav.generate = AsyncMock()
    avail = _make_mock_backend("avail")
    avail.generate = AsyncMock(return_value=GenerateResult("done", 10, "stop", {}, "avail"))
    router = ClassifierRouter(backends=[unav, avail])
    result = await router.generate([{"role": "user", "content": "hi"}])
    assert result.text == "done"
    assert result.backend == "avail"
    unav.generate.assert_not_called()
    avail.generate.assert_called_once()


async def test_router_generate_all_fail_raises():
    a = _make_mock_backend("a")
    a.generate = AsyncMock(side_effect=RuntimeError("err"))
    b = _make_mock_backend("b", available=False)
    router = ClassifierRouter(backends=[a, b])
    with pytest.raises(RuntimeError, match="All backends failed or unavailable"):
        await router.generate([{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# ping_all
# ---------------------------------------------------------------------------

async def test_router_ping_all():
    a = _make_mock_backend("a")
    b = _make_mock_backend("b")
    router = ClassifierRouter(backends=[a, b])
    await router.ping_all()
    a.ping.assert_called_once()
    b.ping.assert_called_once()
