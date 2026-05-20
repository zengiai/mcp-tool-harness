from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def clock() -> dict[str, float]:
    return {"now": 0.0}


@pytest.fixture
def time_func(clock: dict[str, float]):
    return lambda: clock["now"]
