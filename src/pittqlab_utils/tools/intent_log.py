"""
IntentLogger — structured JSONL logging for intent classification observability.

Writes each classification event to ~/.menakai/logs/intent_log.jsonl.
Supports get_recent() and get_summary() for stats queries.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

_DEFAULT_LOG_PATH = Path.home() / ".menakai" / "logs" / "intent_log.jsonl"


def _write_line_sync(log_path: Path, line: str) -> None:
    """Synchronous write to JSONL file. Creates dirs if needed."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _read_all_sync(log_path: Path) -> List[Dict[str, Any]]:
    """Synchronous read of all records from JSONL file."""
    if not log_path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


class IntentLogger:
    """
    Logs intent classification events to a JSONL file.
    Non-blocking writes via asyncio.to_thread.
    """

    def __init__(self, log_path: Optional[Path] = None):
        self._log_path = log_path if log_path is not None else _DEFAULT_LOG_PATH

    async def log(self, event: Dict[str, Any]) -> None:
        """
        Append event to JSONL file. Creates dirs if needed.
        Fire-and-forget via asyncio.to_thread (non-blocking).
        """
        line = json.dumps(event, ensure_ascii=False)
        asyncio.create_task(asyncio.to_thread(_write_line_sync, self._log_path, line))

    async def get_recent(self, n: int = 20) -> List[Dict[str, Any]]:
        """Return last n records from the log file."""
        records = await asyncio.to_thread(_read_all_sync, self._log_path)
        return records[-n:] if len(records) > n else records

    async def get_summary(self) -> Dict[str, Any]:
        """
        Return summary stats: total_calls, label_counts, avg_confidence,
        backend_usage, avg_latency_ms.
        """
        records = await asyncio.to_thread(_read_all_sync, self._log_path)
        if not records:
            return {
                "total_calls": 0,
                "label_counts": {},
                "avg_confidence": 0.0,
                "backend_usage": {},
                "avg_latency_ms": 0.0,
            }

        label_counts: Dict[str, int] = {}
        backend_usage: Dict[str, int] = {}
        total_confidence = 0.0
        total_latency = 0.0
        latency_count = 0

        for r in records:
            label = r.get("final_label") or r.get("label") or "unknown"
            label_counts[label] = label_counts.get(label, 0) + 1

            backends = r.get("backends") or []
            if isinstance(backends, list):
                for b in backends:
                    if isinstance(b, str):
                        backend_usage[b] = backend_usage.get(b, 0) + 1
                    elif isinstance(b, dict) and b.get("backend"):
                        bname = b["backend"]
                        backend_usage[bname] = backend_usage.get(bname, 0) + 1

            conf = r.get("final_confidence") or r.get("confidence")
            if conf is not None:
                total_confidence += float(conf)

            lat = r.get("latency_ms")
            if lat is not None:
                total_latency += float(lat)
                latency_count += 1

        n = len(records)
        return {
            "total_calls": n,
            "label_counts": label_counts,
            "avg_confidence": total_confidence / n if n else 0.0,
            "backend_usage": backend_usage,
            "avg_latency_ms": total_latency / latency_count if latency_count else 0.0,
        }
