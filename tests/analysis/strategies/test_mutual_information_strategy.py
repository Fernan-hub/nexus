"""Tests for MutualInformationStrategy."""

import math
import dataclasses
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import quantities as pq
from neo.core import AnalogSignal

from nexus.analysis.estimators.mutual_information import (
    DiscreteMutualInformationConfig,
    KernelMutualInformationConfig,
)
from nexus.analysis.results.matrix_result import MatrixResult
from nexus.analysis.strategies.mutual_information_strategy import (
    MutualInformationStrategy,
    MutualInformationStrategyDataInput,
    MutualInformationStrategyFilterCriteria,
)
from tests.utils import Expected, Given, Scenario


def _make_signal(name: str | None = None) -> AnalogSignal:
    return AnalogSignal(
        np.array([1.0, 2.0, 3.0]) * pq.mV,
        sampling_rate=1.0 * pq.kHz,
        name=name,
    )


class TestMutualInformationStrategy(unittest.TestCase):
    """Unit tests for MutualInformationStrategy."""

    @pytest.mark.unit
    @pytest.mark.strategy
    @patch("nexus.analysis.estimators.mutual_information.DiscreteMIEstimator")
    def test_run_analysis(self, mock_estimator_class: MagicMock) -> None:
        """Test that run_analysis computes one MI value per (x, y) pair."""
        scenarios = [
            Scenario(
                name="one x one y",
                given=Given(
                    data={
                        "data_x": [_make_signal()],
                        "data_y": [_make_signal()],
                    }
                ),
                expected=Expected(data={"element_count": 1}),
            ),
            Scenario(
                name="two x two y",
                given=Given(
                    data={
                        "data_x": [_make_signal(), _make_signal()],
                        "data_y": [_make_signal(), _make_signal()],
                    }
                ),
                expected=Expected(data={"element_count": 4}),
            ),
        ]

        for scenario in scenarios:
            with self.subTest(msg=scenario.name):
                mock_estimator_class.reset_mock()
                mock_estimator_class.return_value.result.return_value = 0.5

                strategy = MutualInformationStrategy(DiscreteMutualInformationConfig())
                data_input = MutualInformationStrategyDataInput(
                    data_x=scenario.given.data["data_x"],
                    data_y=scenario.given.data["data_y"],
                )

                result = strategy.run_analysis(data_input)

                self.assertIsInstance(result, MatrixResult)
                self.assertEqual(result.algorithm, "mutual_information")
                self.assertEqual(
                    len(result.matrix), scenario.expected.data["element_count"]
                )
                self.assertEqual(
                    mock_estimator_class.call_count,
                    scenario.expected.data["element_count"],
                )

    @pytest.mark.unit
    @pytest.mark.strategy
    @patch("nexus.analysis.estimators.mutual_information.DiscreteMIEstimator")
    def test_run_analysis_labels(self, mock_estimator_class: MagicMock) -> None:
        """Test that signal names are used as labels, falling back to index when absent."""
        scenarios = [
            Scenario(
                name="named signals",
                given=Given(
                    data={
                        "data_x": _make_signal(name="signal_a"),
                        "data_y": _make_signal(name="signal_b"),
                    }
                ),
                expected=Expected(data={"row": "signal_a", "column": "signal_b"}),
            ),
            Scenario(
                name="unnamed signals",
                given=Given(
                    data={
                        "data_x": _make_signal(name=None),
                        "data_y": _make_signal(name=None),
                    }
                ),
                expected=Expected(data={"row": "data_x_0", "column": "data_y_0"}),
            ),
        ]

        for scenario in scenarios:
            with self.subTest(msg=scenario.name):
                mock_estimator_class.reset_mock()
                mock_estimator_class.return_value.result.return_value = 0.5

                strategy = MutualInformationStrategy(DiscreteMutualInformationConfig())
                data_input = MutualInformationStrategyDataInput(
                    data_x=[scenario.given.data["data_x"]],
                    data_y=[scenario.given.data["data_y"]],
                )

                result = strategy.run_analysis(data_input)

                self.assertEqual(result.matrix[0].row, scenario.expected.data["row"])
                self.assertEqual(
                    result.matrix[0].column, scenario.expected.data["column"]
                )

    @pytest.mark.unit
    @pytest.mark.strategy
    @patch("nexus.analysis.estimators.mutual_information.DiscreteMIEstimator")
    def test_run_analysis_result_values_come_from_estimator(
        self, mock_estimator_class: MagicMock
    ) -> None:
        """Test that element values in the matrix come from the estimator."""
        mock_estimator_class.return_value.result.side_effect = [0.3, 0.7, 0.1, 0.9]

        strategy = MutualInformationStrategy(DiscreteMutualInformationConfig())
        data_input = MutualInformationStrategyDataInput(
            data_x=[_make_signal(), _make_signal()],
            data_y=[_make_signal(), _make_signal()],
        )

        result = strategy.run_analysis(data_input)

        values = [el.value for el in result.matrix]
        self.assertAlmostEqual(values[0], 0.3)
        self.assertAlmostEqual(values[1], 0.7)
        self.assertAlmostEqual(values[2], 0.1)
        self.assertAlmostEqual(values[3], 0.9)


    @pytest.mark.unit
    @pytest.mark.strategy
    def test_filter_criteria_type(self) -> None:
        strategy = MutualInformationStrategy(DiscreteMutualInformationConfig())

        self.assertIs(strategy.filter_criteria_type, MutualInformationStrategyFilterCriteria)

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_data_input_type(self) -> None:
        strategy = MutualInformationStrategy(DiscreteMutualInformationConfig())

        self.assertIs(strategy.data_input_type, MutualInformationStrategyDataInput)

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_input_types_share_field_names(self) -> None:
        strategy = MutualInformationStrategy(DiscreteMutualInformationConfig())
        filter_fields = {f.name for f in dataclasses.fields(strategy.filter_criteria_type)}
        data_fields = {f.name for f in dataclasses.fields(strategy.data_input_type)}

        self.assertEqual(filter_fields, data_fields)


class TestIntegrationMutualInformationStrategy(unittest.TestCase):
    """Integration tests for MutualInformationStrategy against real infomeasure estimators."""

    @pytest.mark.integration
    @pytest.mark.strategy
    def test_mi_of_signal_with_itself_equals_entropy(self) -> None:
        """MI(X, X) = H(X): mutual information of a signal with itself equals its entropy."""
        data = np.tile([0.0, 1.0], 50)
        sig = AnalogSignal(data * pq.mV, sampling_rate=1.0 * pq.kHz)

        result = MutualInformationStrategy(
            DiscreteMutualInformationConfig()
        ).run_analysis(MutualInformationStrategyDataInput(data_x=[sig], data_y=[sig]))

        self.assertAlmostEqual(result.matrix[0].value, np.log(2), places=10)

    @pytest.mark.integration
    @pytest.mark.strategy
    def test_mi_of_independent_signals_is_negligible(self) -> None:
        """MI(X, Y) ~= 0 when X and Y are drawn independently."""
        rng = np.random.default_rng(42)
        sig_x = AnalogSignal(
            rng.integers(0, 2, 1000).astype(float) * pq.mV, sampling_rate=1.0 * pq.kHz
        )
        sig_y = AnalogSignal(
            rng.integers(0, 2, 1000).astype(float) * pq.mV, sampling_rate=1.0 * pq.kHz
        )

        result = MutualInformationStrategy(
            DiscreteMutualInformationConfig()
        ).run_analysis(
            MutualInformationStrategyDataInput(data_x=[sig_x], data_y=[sig_y])
        )

        self.assertLess(abs(result.matrix[0].value), 0.01)

    @pytest.mark.integration
    @pytest.mark.strategy
    def test_conditional_mi_of_signal_with_itself_given_itself_is_zero(self) -> None:
        """CMI(X; X | X) = 0: knowing X makes Y=X and X itself fully redundant."""
        data = np.tile([0.0, 1.0], 50)
        sig = AnalogSignal(data * pq.mV, sampling_rate=1.0 * pq.kHz)

        result = MutualInformationStrategy(
            DiscreteMutualInformationConfig()
        ).run_analysis(
            MutualInformationStrategyDataInput(data_x=[sig], data_y=[sig], cond=[sig])
        )

        self.assertAlmostEqual(result.matrix[0].value, 0.0, places=10)

    @pytest.mark.integration
    @pytest.mark.strategy
    @pytest.mark.slow
    def test_kernel_mi_produces_finite_result(self) -> None:
        """KernelMutualInformationConfig wires up correctly: the strategy runs without error and returns a finite value."""
        rng = np.random.default_rng(1)
        sig_x = AnalogSignal(rng.normal(0, 1, 100) * pq.mV, sampling_rate=1.0 * pq.kHz)
        sig_y = AnalogSignal(rng.normal(0, 1, 100) * pq.mV, sampling_rate=1.0 * pq.kHz)

        result = MutualInformationStrategy(
            KernelMutualInformationConfig()
        ).run_analysis(
            MutualInformationStrategyDataInput(data_x=[sig_x], data_y=[sig_y])
        )

        self.assertIsInstance(result, MatrixResult)
        self.assertTrue(math.isfinite(result.matrix[0].value))

    @pytest.mark.integration
    @pytest.mark.strategy
    @pytest.mark.slow
    def test_kernel_conditional_mi_produces_finite_result(self) -> None:
        """KernelMutualInformationConfig conditional path wires up correctly and returns a finite value."""
        rng = np.random.default_rng(2)
        sig_x = AnalogSignal(rng.normal(0, 1, 100) * pq.mV, sampling_rate=1.0 * pq.kHz)
        sig_y = AnalogSignal(rng.normal(0, 1, 100) * pq.mV, sampling_rate=1.0 * pq.kHz)
        cond = AnalogSignal(rng.normal(0, 1, 100) * pq.mV, sampling_rate=1.0 * pq.kHz)

        result = MutualInformationStrategy(
            KernelMutualInformationConfig()
        ).run_analysis(
            MutualInformationStrategyDataInput(
                data_x=[sig_x], data_y=[sig_y], cond=[cond]
            )
        )

        self.assertIsInstance(result, MatrixResult)
        self.assertTrue(math.isfinite(result.matrix[0].value))
