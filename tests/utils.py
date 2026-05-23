"""Shared test utilities for scenario-based testing."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Given:
    """Configuration for the Arrange phase of a test scenario."""
    data: dict[str, Any] = field(default_factory=dict)
    return_values: dict[str, Any] = field(default_factory=dict)
    side_effects: dict[str, Any] = field(default_factory=dict)


@dataclass
class Expected:
    """Configuration for the Assert phase of a test scenario."""
    data: dict[str, Any] = field(default_factory=dict)
    exceptions: dict[str, Any] = field(default_factory=dict)
    calls: dict[str, Any] = field(default_factory=dict)


@dataclass
class Scenario:
    """A single test scenario combining Given and Expected configurations."""
    name: str
    given: Given
    expected: Expected
