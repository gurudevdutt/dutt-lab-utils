"""Tests for IntentLogger. All tests use tmp_path, never write to ~/.menakai."""

import asyncio
import json
import pytest
from pathlib import Path

from pittqlab_utils.tools.intent_log import IntentLogger


@pytest.fixture
def logger(tmp_path):
    """IntentLogger with tmp_path log file."""
    log_path = tmp_path / "intent_log.jsonl"
    return IntentLogger(log_path=log_path)


@pytest.mark.asyncio
async def test_log_writes_valid_json_to_file(logger, tmp_path):
    """log() writes valid JSON to file."""
    event = {
        "timestamp": "2025-03-19T12:00:00Z",
        "normalized_text": "remind me to follow up",
        "keyword_hit": "reminder",
        "final_label": "reminder",
        "final_confidence": 1.0,
        "reason": "keyword_hit_explicit",
        "backends": [],
        "total_tokens": 0,
        "latency_ms": 5.2,
    }
    await logger.log(event)
    # Fire-and-forget: allow thread pool to complete
    await asyncio.sleep(0.05)

    log_path = tmp_path / "intent_log.jsonl"
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["final_label"] == "reminder"
    assert parsed["normalized_text"] == "remind me to follow up"
    assert parsed["latency_ms"] == 5.2


@pytest.mark.asyncio
async def test_get_recent_returns_correct_number_of_records(logger):
    """get_recent() returns correct number of records."""
    for i in range(25):
        await logger.log({
            "timestamp": f"2025-03-19T12:{i:02d}:00Z",
            "normalized_text": f"msg {i}",
            "keyword_hit": None,
            "final_label": "research",
            "final_confidence": 0.9,
            "reason": "consensus",
            "backends": ["gemini"],
            "total_tokens": 10,
            "latency_ms": 50.0,
        })
    # Fire-and-forget: allow all 25 writes to complete
    await asyncio.sleep(0.25)

    recent = await logger.get_recent(n=10)
    assert len(recent) == 10

    recent_all = await logger.get_recent(n=100)
    assert len(recent_all) == 25


@pytest.mark.asyncio
async def test_get_summary_returns_correct_label_counts(logger):
    """get_summary() returns correct label counts."""
    events = [
        {"final_label": "research", "final_confidence": 0.9, "backends": ["gemini"], "latency_ms": 50.0},
        {"final_label": "research", "final_confidence": 0.85, "backends": ["gemini"], "latency_ms": 45.0},
        {"final_label": "note", "final_confidence": 1.0, "backends": [], "latency_ms": 2.0},
        {"final_label": "chitchat", "final_confidence": 1.0, "backends": [], "latency_ms": 1.0},
    ]
    for e in events:
        await logger.log({
            "timestamp": "2025-03-19T12:00:00Z",
            "normalized_text": "test",
            "keyword_hit": None,
            "reason": "test",
            **e,
        })
    await asyncio.sleep(0.1)

    summary = await logger.get_summary()
    assert summary["total_calls"] == 4
    assert summary["label_counts"] == {"research": 2, "note": 1, "chitchat": 1}
    assert 0.9 < summary["avg_confidence"] < 1.0
    assert summary["backend_usage"]["gemini"] == 2
    assert summary["avg_latency_ms"] > 0


@pytest.mark.asyncio
async def test_get_recent_empty_file(logger):
    """get_recent() on empty/missing file returns empty list."""
    recent = await logger.get_recent(n=20)
    assert recent == []


@pytest.mark.asyncio
async def test_get_summary_empty_file(logger):
    """get_summary() on empty file returns zeros."""
    summary = await logger.get_summary()
    assert summary["total_calls"] == 0
    assert summary["label_counts"] == {}
    assert summary["avg_confidence"] == 0.0
    assert summary["backend_usage"] == {}
    assert summary["avg_latency_ms"] == 0.0
