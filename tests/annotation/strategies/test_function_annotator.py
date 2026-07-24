"""Tests for FunctionAnnotator."""

import unittest
from dataclasses import dataclass, field

import pytest

from nexus.annotation.strategies.function_annotator import FunctionAnnotator
from nexus.protocols import AnnotatedItem
from nexus.types import Annotation
from tests.utils import Expected, Given, Scenario


@dataclass
class _AnnotatedStub:
    annotations: Annotation = field(default_factory=dict)


class TestFunctionAnnotator(unittest.TestCase):
    """Unit tests for FunctionAnnotator."""

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_annotate(self) -> None:
        """Test that annotate applies ann_func to each item and updates annotations."""
        scenarios = [
            Scenario(
                name="single item with fixed annotation",
                given=Given(
                    data={
                        "items": [_AnnotatedStub()],
                        "ann_func": lambda item: {"label": "fixed"},
                    }
                ),
                expected=Expected(
                    data={"annotations": [{"label": "fixed"}]},
                ),
            ),
            Scenario(
                name="multiple items receive same annotation",
                given=Given(
                    data={
                        "items": [_AnnotatedStub(), _AnnotatedStub()],
                        "ann_func": lambda item: {"type": "signal"},
                    }
                ),
                expected=Expected(
                    data={"annotations": [{"type": "signal"}, {"type": "signal"}]},
                ),
            ),
            Scenario(
                name="annotation derived from existing item annotations",
                given=Given(
                    data={
                        "items": [
                            _AnnotatedStub(annotations={"model": "A"}),
                            _AnnotatedStub(annotations={"model": "B"}),
                        ],
                        "ann_func": lambda item: {
                            "derived": item.annotations["model"].lower()
                        },
                    }
                ),
                expected=Expected(
                    data={
                        "annotations": [
                            {"model": "A", "derived": "a"},
                            {"model": "B", "derived": "b"},
                        ]
                    },
                ),
            ),
        ]

        for scenario in scenarios:
            with self.subTest(msg=scenario.name):
                strategy = FunctionAnnotator(ann_func=scenario.given.data["ann_func"])

                strategy.annotate(scenario.given.data["items"])

                for item, expected_ann in zip(
                    scenario.given.data["items"],
                    scenario.expected.data["annotations"],
                ):
                    self.assertEqual(item.annotations, expected_ann)

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_annotate_raises_runtime_error_when_function_raises(self) -> None:
        """Test that annotate wraps ann_func exceptions in RuntimeError."""

        def failing_func(item: AnnotatedItem) -> Annotation:
            raise ValueError("bad item")

        strategy = FunctionAnnotator(ann_func=failing_func)

        with self.assertRaises(RuntimeError):
            strategy.annotate([_AnnotatedStub()])
