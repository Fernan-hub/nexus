"""Unit tests for entropy estimator configs."""

import unittest

import pytest
from infomeasure.estimators.entropy import (
    DiscreteEntropyEstimator,
    KernelEntropyEstimator,
)

from nexus.analysis.constants import KernelType
from nexus.analysis.estimators.entropy import DiscreteEntropyConfig, KernelEntropyConfig
from tests.utils import Expected, Given, Scenario


class TestDiscreteEntropyConfig(unittest.TestCase):
    """Unit tests for DiscreteEntropyConfig."""

    @pytest.mark.unit
    def test_get_estimator_class(self) -> None:
        self.assertIs(
            DiscreteEntropyConfig().get_estimator_class(), DiscreteEntropyEstimator
        )

    @pytest.mark.unit
    def test_get_estimator_kwargs(self) -> None:
        self.assertEqual(DiscreteEntropyConfig().get_estimator_kwargs(), {})

    @pytest.mark.unit
    def test_get_conditional_estimator_class_raises(self) -> None:
        with self.assertRaises(NotImplementedError):
            DiscreteEntropyConfig().get_conditional_estimator_class()


class TestKernelEntropyConfig(unittest.TestCase):
    """Unit tests for KernelEntropyConfig."""

    @pytest.mark.unit
    def test_get_estimator_class(self) -> None:
        self.assertIs(
            KernelEntropyConfig().get_estimator_class(), KernelEntropyEstimator
        )

    @pytest.mark.unit
    def test_get_estimator_kwargs(self) -> None:
        scenarios = [
            Scenario(
                name="default",
                given=Given(data={"config": KernelEntropyConfig()}),
                expected=Expected(
                    data={
                        "kwargs": {"bandwidth": 1.0, "kernel": "gaussian", "workers": 1}
                    }
                ),
            ),
            Scenario(
                name="custom",
                given=Given(
                    data={
                        "config": KernelEntropyConfig(
                            bandwidth=2.0, kernel=KernelType.BOX, workers=4
                        )
                    }
                ),
                expected=Expected(
                    data={"kwargs": {"bandwidth": 2.0, "kernel": "box", "workers": 4}}
                ),
            ),
        ]

        for scenario in scenarios:
            with self.subTest(msg=scenario.name):
                self.assertEqual(
                    scenario.given.data["config"].get_estimator_kwargs(),
                    scenario.expected.data["kwargs"],
                )

    @pytest.mark.unit
    def test_get_estimator_kwargs_kernel_is_string_not_enum(self) -> None:
        """kernel must be a plain string, not a KernelType enum, for infomeasure compatibility."""
        kwargs = KernelEntropyConfig().get_estimator_kwargs()
        self.assertIsInstance(kwargs["kernel"], str)

    @pytest.mark.unit
    def test_get_conditional_estimator_class_raises(self) -> None:
        with self.assertRaises(NotImplementedError):
            KernelEntropyConfig().get_conditional_estimator_class()
