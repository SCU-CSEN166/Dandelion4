"""Tests for portable CPU resource detection."""

import os

import pytest

from src.core import runtime


def test_available_cpu_count_uses_affinity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 2, 4}, raising=False)
    assert runtime.available_cpu_count() == 3


def test_available_cpu_count_falls_back_when_affinity_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(pid: int) -> set[int]:
        raise OSError("affinity unavailable")

    monkeypatch.setattr(os, "sched_getaffinity", unavailable, raising=False)
    monkeypatch.setattr(os, "process_cpu_count", lambda: 5, raising=False)
    assert runtime.available_cpu_count() == 5
