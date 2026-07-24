"""Tests for GeneratorAnnotator."""

import unittest
from dataclasses import dataclass, field
from typing import Generator

import pytest

from nexus.annotation.strategies.generator_annotator import GeneratorAnnotator
from nexus.types import Annotation
from tests.utils import Expected, Given, Scenario


@dataclass
class _AnnotatedStub:
    annotations: Annotation = field(default_factory=dict)


def _make_gen(annotations: list[Annotation]) -> Generator[Annotation, None, None]:
    yield from annotations


class TestGeneratorAnnotator(unittest.TestCase):
    """Unit tests for GeneratorAnnotator."""

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_annotate(self) -> None:
        """Test that annotate consumes annotations from the generator for each item."""
        scenarios = [
            Scenario(
                name="generator yields unique annotation per item",
                given=Given(
                    data={
                        "items": [_AnnotatedStub(), _AnnotatedStub(), _AnnotatedStub()],
                        "gen_annotations": [
                            {"label": "first"},
                            {"label": "second"},
                            {"label": "third"},
                        ],
                        "default": None,
                    }
                ),
                expected=Expected(
                    data={
                        "annotations": [
                            {"label": "first"},
                            {"label": "second"},
                            {"label": "third"},
                        ]
                    },
                ),
            ),
            Scenario(
                name="exhausted generator falls back to default annotation",
                given=Given(
                    data={
                        "items": [_AnnotatedStub(), _AnnotatedStub(), _AnnotatedStub()],
                        "gen_annotations": [{"label": "first"}],
                        "default": {"label": "fallback"},
                    }
                ),
                expected=Expected(
                    data={
                        "annotations": [
                            {"label": "first"},
                            {"label": "fallback"},
                            {"label": "fallback"},
                        ]
                    },
                ),
            ),
        ]

        for scenario in scenarios:
            with self.subTest(msg=scenario.name):
                gen = _make_gen(scenario.given.data["gen_annotations"])
                strategy = GeneratorAnnotator(
                    ann_gen=gen,
                    default_annotation=scenario.given.data["default"],
                )

                strategy.annotate(scenario.given.data["items"])

                for item, expected_ann in zip(
                    scenario.given.data["items"],
                    scenario.expected.data["annotations"],
                ):
                    self.assertEqual(item.annotations, expected_ann)

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_annotate_raises_runtime_error_when_generator_exhausted_without_default(
        self,
    ) -> None:
        """Test annotate raises RuntimeError: generator exhausted with no default."""
        gen = _make_gen([])
        strategy = GeneratorAnnotator(ann_gen=gen)

        with self.assertRaises(RuntimeError):
            strategy.annotate([_AnnotatedStub()])
