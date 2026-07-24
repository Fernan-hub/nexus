"""Tests for TransferEntropyStrategy."""

import math
import dataclasses
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import quantities as pq
from neo.core import AnalogSignal

from nexus.analysis.estimators.transfer_entropy import (
    DiscreteTransferEntropyConfig,
    KernelTransferEntropyConfig,
)
from nexus.analysis.results.matrix_result import MatrixResult
from nexus.analysis.strategies.transfer_entropy_strategy import (
    TransferEntropyStrategy,
    TransferEntropyStrategyDataInput,
    TransferEntropyStrategyFilterCriteria,
)
from tests.utils import Expected, Given, Scenario


def _make_signal(name: str | None = None) -> AnalogSignal:
    return AnalogSignal(
        np.array([1.0, 2.0, 3.0]) * pq.mV,
        sampling_rate=1.0 * pq.kHz,
        name=name,
    )


class TestTransferEntropyStrategy(unittest.TestCase):
    """Unit tests for TransferEntropyStrategy."""

    @pytest.mark.unit
    @pytest.mark.strategy
    @patch("nexus.analysis.estimators.transfer_entropy.DiscreteTEEstimator")
    def test_run_analysis(self, mock_estimator_class: MagicMock) -> None:
        """Test that run_analysis computes one TE value per (source, dest) pair."""
        scenarios = [
            Scenario(
                name="one source one dest",
                given=Given(
                    data={
                        "sources": [_make_signal()],
                        "dests": [_make_signal()],
                    }
                ),
                expected=Expected(data={"element_count": 1}),
            ),
            Scenario(
                name="two sources two dests",
                given=Given(
                    data={
                        "sources": [_make_signal(), _make_signal()],
                        "dests": [_make_signal(), _make_signal()],
                    }
                ),
                expected=Expected(data={"element_count": 4}),
            ),
        ]

        for scenario in scenarios:
            with self.subTest(msg=scenario.name):
                mock_estimator_class.reset_mock()
                mock_estimator_class.return_value.result.return_value = 0.5

                strategy = TransferEntropyStrategy(DiscreteTransferEntropyConfig())
                data_input = TransferEntropyStrategyDataInput(
                    sources=scenario.given.data["sources"],
                    dests=scenario.given.data["dests"],
                )

                result = strategy.run_analysis(data_input)

                self.assertIsInstance(result, MatrixResult)
                self.assertEqual(result.algorithm, "transfer_entropy")
                self.assertEqual(
                    len(result.matrix), scenario.expected.data["element_count"]
                )
                self.assertEqual(
                    mock_estimator_class.call_count,
                    scenario.expected.data["element_count"],
                )

    @pytest.mark.unit
    @pytest.mark.strategy
    @patch("nexus.analysis.estimators.transfer_entropy.DiscreteTEEstimator")
    def test_run_analysis_labels(self, mock_estimator_class: MagicMock) -> None:
        """Test signal names are used as labels, falling back to index when absent."""
        scenarios = [
            Scenario(
                name="named signals",
                given=Given(
                    data={
                        "source": _make_signal(name="neuron_a"),
                        "dest": _make_signal(name="neuron_b"),
                    }
                ),
                expected=Expected(data={"row": "neuron_a", "column": "neuron_b"}),
            ),
            Scenario(
                name="unnamed signals",
                given=Given(
                    data={
                        "source": _make_signal(name=None),
                        "dest": _make_signal(name=None),
                    }
                ),
                expected=Expected(data={"row": "source_0", "column": "dest_0"}),
            ),
        ]

        for scenario in scenarios:
            with self.subTest(msg=scenario.name):
                mock_estimator_class.reset_mock()
                mock_estimator_class.return_value.result.return_value = 0.5

                strategy = TransferEntropyStrategy(DiscreteTransferEntropyConfig())
                data_input = TransferEntropyStrategyDataInput(
                    sources=[scenario.given.data["source"]],
                    dests=[scenario.given.data["dest"]],
                )

                result = strategy.run_analysis(data_input)

                self.assertEqual(result.matrix[0].row, scenario.expected.data["row"])
                self.assertEqual(
                    result.matrix[0].column, scenario.expected.data["column"]
                )

    @pytest.mark.unit
    @pytest.mark.strategy
    @patch("nexus.analysis.estimators.transfer_entropy.DiscreteTEEstimator")
    def test_run_analysis_result_values_come_from_estimator(
        self, mock_estimator_class: MagicMock
    ) -> None:
        """Test that element values in the matrix come from the estimator."""
        mock_estimator_class.return_value.result.side_effect = [0.2, 0.8]

        strategy = TransferEntropyStrategy(DiscreteTransferEntropyConfig())
        data_input = TransferEntropyStrategyDataInput(
            sources=[_make_signal(), _make_signal()],
            dests=[_make_signal()],
        )

        result = strategy.run_analysis(data_input)

        values = [el.value for el in result.matrix]
        self.assertAlmostEqual(values[0], 0.2)
        self.assertAlmostEqual(values[1], 0.8)

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_filter_criteria_type(self) -> None:
        """Returns TransferEntropyStrategyFilterCriteria as filter_criteria_type."""
        strategy = TransferEntropyStrategy(DiscreteTransferEntropyConfig())

        self.assertIs(
            strategy.filter_criteria_type, TransferEntropyStrategyFilterCriteria
        )

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_data_input_type(self) -> None:
        """Returns TransferEntropyStrategyDataInput as data_input_type."""
        strategy = TransferEntropyStrategy(DiscreteTransferEntropyConfig())

        self.assertIs(strategy.data_input_type, TransferEntropyStrategyDataInput)

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_input_types_share_field_names(self) -> None:
        """filter_criteria_type and data_input_type share identical field names."""
        strategy = TransferEntropyStrategy(DiscreteTransferEntropyConfig())
        filter_fields = {
            f.name for f in dataclasses.fields(strategy.filter_criteria_type)
        }
        data_fields = {f.name for f in dataclasses.fields(strategy.data_input_type)}

        self.assertEqual(filter_fields, data_fields)


class TestIntegrationTransferEntropyStrategy(unittest.TestCase):
    """Integration tests for TransferEntropyStrategy against real estimators."""

    @pytest.mark.integration
    @pytest.mark.strategy
    def test_te_detects_causal_dependency(self) -> None:
        """TE(X->Y) ~= ln(2) nats when Y[t] = X[t-1] (perfect causal shift)."""
        rng = np.random.default_rng(42)
        x = rng.integers(0, 2, 500).astype(float)
        y = np.roll(x, 1)
        sig_x = AnalogSignal(x * pq.mV, sampling_rate=1.0 * pq.kHz)
        sig_y = AnalogSignal(y * pq.mV, sampling_rate=1.0 * pq.kHz)

        result = TransferEntropyStrategy(DiscreteTransferEntropyConfig()).run_analysis(
            TransferEntropyStrategyDataInput(sources=[sig_x], dests=[sig_y])
        )

        self.assertAlmostEqual(result.matrix[0].value, np.log(2), delta=0.05)

    @pytest.mark.integration
    @pytest.mark.strategy
    def test_te_of_independent_signals_is_negligible(self) -> None:
        """TE(X -> Y) ~= 0 when X and Y are drawn independently."""
        rng = np.random.default_rng(0)
        sig_x = AnalogSignal(
            rng.integers(0, 2, 500).astype(float) * pq.mV, sampling_rate=1.0 * pq.kHz
        )
        sig_y = AnalogSignal(
            rng.integers(0, 2, 500).astype(float) * pq.mV, sampling_rate=1.0 * pq.kHz
        )

        result = TransferEntropyStrategy(DiscreteTransferEntropyConfig()).run_analysis(
            TransferEntropyStrategyDataInput(sources=[sig_x], dests=[sig_y])
        )

        self.assertLess(abs(result.matrix[0].value), 0.01)

    @pytest.mark.integration
    @pytest.mark.strategy
    def test_conditional_te_detects_causal_dependency_given_independent_signal(
        self,
    ) -> None:
        """CTE(X->Y|Z) ~= ln(2) nats: independent Z does not suppress detected TE."""
        rng = np.random.default_rng(42)
        x = rng.integers(0, 2, 500).astype(float)
        y = np.roll(x, 1)
        sig_x = AnalogSignal(x * pq.mV, sampling_rate=1.0 * pq.kHz)
        sig_y = AnalogSignal(y * pq.mV, sampling_rate=1.0 * pq.kHz)
        cond = AnalogSignal(
            rng.integers(0, 2, 500).astype(float) * pq.mV, sampling_rate=1.0 * pq.kHz
        )

        result = TransferEntropyStrategy(DiscreteTransferEntropyConfig()).run_analysis(
            TransferEntropyStrategyDataInput(
                sources=[sig_x], dests=[sig_y], cond=[cond]
            )
        )

        self.assertAlmostEqual(result.matrix[0].value, np.log(2), delta=0.05)

    @pytest.mark.integration
    @pytest.mark.strategy
    @pytest.mark.slow
    def test_kernel_te_produces_finite_result(self) -> None:
        """KernelTransferEntropyConfig wires up and returns a finite value."""
        rng = np.random.default_rng(1)
        sig_x = AnalogSignal(rng.normal(0, 1, 100) * pq.mV, sampling_rate=1.0 * pq.kHz)
        sig_y = AnalogSignal(rng.normal(0, 1, 100) * pq.mV, sampling_rate=1.0 * pq.kHz)

        result = TransferEntropyStrategy(KernelTransferEntropyConfig()).run_analysis(
            TransferEntropyStrategyDataInput(sources=[sig_x], dests=[sig_y])
        )

        self.assertIsInstance(result, MatrixResult)
        self.assertTrue(math.isfinite(result.matrix[0].value))

    @pytest.mark.integration
    @pytest.mark.strategy
    @pytest.mark.slow
    def test_kernel_conditional_te_produces_finite_result(self) -> None:
        """KernelTE conditional config wires up and returns a finite value."""
        rng = np.random.default_rng(2)
        sig_x = AnalogSignal(rng.normal(0, 1, 100) * pq.mV, sampling_rate=1.0 * pq.kHz)
        sig_y = AnalogSignal(rng.normal(0, 1, 100) * pq.mV, sampling_rate=1.0 * pq.kHz)
        cond = AnalogSignal(rng.normal(0, 1, 100) * pq.mV, sampling_rate=1.0 * pq.kHz)

        result = TransferEntropyStrategy(KernelTransferEntropyConfig()).run_analysis(
            TransferEntropyStrategyDataInput(
                sources=[sig_x], dests=[sig_y], cond=[cond]
            )
        )

        self.assertIsInstance(result, MatrixResult)
        self.assertTrue(math.isfinite(result.matrix[0].value))
