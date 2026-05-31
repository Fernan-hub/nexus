"""Tests for JointEntropyStrategy."""

import dataclasses
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import quantities as pq
from neo.core import AnalogSignal

from nexus.analysis.estimators.entropy import DiscreteEntropyConfig
from nexus.analysis.results.scalar_result import ScalarResult
from nexus.analysis.strategies.joint_entropy_strategy import (
    JointEntropyStrategy,
    JointEntropyStrategyDataInput,
    JointEntropyStrategyFilterCriteria,
)
from tests.utils import Expected, Given, Scenario


def _make_signal(name: str | None = None) -> AnalogSignal:
    return AnalogSignal(
        np.array([1.0, 2.0, 3.0]) * pq.mV,
        sampling_rate=1.0 * pq.kHz,
        name=name,
    )


class TestJointEntropyStrategy(unittest.TestCase):
    """Unit tests for JointEntropyStrategy."""

    @pytest.mark.unit
    @pytest.mark.strategy
    @patch("nexus.analysis.estimators.entropy.DiscreteEntropyEstimator")
    def test_run_analysis(self, mock_estimator_class: MagicMock) -> None:
        """Test that run_analysis calls the estimator once and returns a ScalarResult."""
        scenarios = [
            Scenario(
                name="two signals",
                given=Given(
                    data={"signals": [_make_signal(name="a"), _make_signal(name="b")]}
                ),
                expected=Expected(
                    data={"label": "Joint entropy of a, b", "value": 0.5}
                ),
            ),
            Scenario(
                name="three signals",
                given=Given(
                    data={
                        "signals": [
                            _make_signal(name="x"),
                            _make_signal(name="y"),
                            _make_signal(name="z"),
                        ]
                    }
                ),
                expected=Expected(
                    data={"label": "Joint entropy of x, y, z", "value": 0.5}
                ),
            ),
        ]

        for scenario in scenarios:
            with self.subTest(msg=scenario.name):
                mock_estimator_class.reset_mock()
                mock_estimator_class.return_value.result.return_value = (
                    scenario.expected.data["value"]
                )

                strategy = JointEntropyStrategy(DiscreteEntropyConfig())
                data_input = JointEntropyStrategyDataInput(
                    data=scenario.given.data["signals"]
                )

                result = strategy.run_analysis(data_input)

                self.assertIsInstance(result, ScalarResult)
                self.assertEqual(result.algorithm, "joint_entropy")
                self.assertAlmostEqual(result.value, scenario.expected.data["value"])
                self.assertEqual(result.label, scenario.expected.data["label"])
                mock_estimator_class.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.strategy
    @patch("nexus.analysis.estimators.entropy.DiscreteEntropyEstimator")
    def test_run_analysis_uses_index_label_for_unnamed_signals(
        self, mock_estimator_class: MagicMock
    ) -> None:
        """Test that signals without a name fall back to index-based labels."""
        mock_estimator_class.return_value.result.return_value = 0.5

        strategy = JointEntropyStrategy(DiscreteEntropyConfig())
        data_input = JointEntropyStrategyDataInput(
            data=[_make_signal(name=None), _make_signal(name=None)]
        )

        result = strategy.run_analysis(data_input)

        self.assertEqual(result.label, "Joint entropy of signal_0, signal_1")

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_filter_criteria_type(self) -> None:
        """Returns JointEntropyStrategyFilterCriteria as filter_criteria_type."""
        strategy = JointEntropyStrategy(DiscreteEntropyConfig())

        self.assertIs(strategy.filter_criteria_type, JointEntropyStrategyFilterCriteria)

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_data_input_type(self) -> None:
        """Returns JointEntropyStrategyDataInput as data_input_type."""
        strategy = JointEntropyStrategy(DiscreteEntropyConfig())

        self.assertIs(strategy.data_input_type, JointEntropyStrategyDataInput)

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_input_types_share_field_names(self) -> None:
        """filter_criteria_type and data_input_type share identical field names."""
        strategy = JointEntropyStrategy(DiscreteEntropyConfig())
        filter_fields = {
            f.name for f in dataclasses.fields(strategy.filter_criteria_type)
        }
        data_fields = {f.name for f in dataclasses.fields(strategy.data_input_type)}

        self.assertEqual(filter_fields, data_fields)


class TestIntegrationJointEntropyStrategy(unittest.TestCase):
    """Integration tests for JointEntropyStrategy against real infomeasure estimators."""

    @pytest.mark.integration
    @pytest.mark.strategy
    def test_joint_entropy_of_independent_signals_equals_sum_of_marginals(self) -> None:
        """H(X, Y) = H(X) + H(Y) when X and Y are independent (entropy additivity)."""
        data_x = np.tile([0.0, 1.0], 50)
        data_y = np.tile([0.0, 0.0, 1.0, 1.0], 25)
        sig_x = AnalogSignal(data_x * pq.mV, sampling_rate=1.0 * pq.kHz)
        sig_y = AnalogSignal(data_y * pq.mV, sampling_rate=1.0 * pq.kHz)

        result = JointEntropyStrategy(DiscreteEntropyConfig()).run_analysis(
            JointEntropyStrategyDataInput(data=[sig_x, sig_y])
        )

        self.assertAlmostEqual(result.value, 2 * np.log(2), places=10)

    @pytest.mark.integration
    @pytest.mark.strategy
    def test_joint_entropy_of_identical_signals_equals_marginal_entropy(self) -> None:
        """H(X, X) = H(X): adding a redundant copy does not increase joint entropy."""
        data = np.tile([0.0, 1.0], 50)
        sig = AnalogSignal(data * pq.mV, sampling_rate=1.0 * pq.kHz)

        result = JointEntropyStrategy(DiscreteEntropyConfig()).run_analysis(
            JointEntropyStrategyDataInput(data=[sig, sig])
        )

        self.assertAlmostEqual(result.value, np.log(2), places=10)
