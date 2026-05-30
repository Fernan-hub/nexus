"""Tests for KLDStrategy."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import quantities as pq
from neo.core import AnalogSignal

from nexus.analysis.estimators.entropy import DiscreteEntropyConfig
from nexus.analysis.results.matrix_result import MatrixResult
from nexus.analysis.strategies.kld_strategy import (
    KLDStrategy,
    KLDStrategyDataInput,
)
from tests.utils import Expected, Given, Scenario


def _make_signal(name: str | None = None) -> AnalogSignal:
    return AnalogSignal(
        np.array([1.0, 2.0, 3.0]) * pq.mV,
        sampling_rate=1.0 * pq.kHz,
        name=name,
    )


class TestKLDStrategy(unittest.TestCase):
    """Unit tests for KLDStrategy."""

    @pytest.mark.unit
    @pytest.mark.strategy
    @patch("nexus.analysis.strategies.kld_strategy.im")
    @patch("nexus.analysis.estimators.entropy.DiscreteEntropyEstimator")
    def test_run_analysis(
        self, _mock_estimator_class: MagicMock, mock_im: MagicMock
    ) -> None:
        """Test that run_analysis computes one KLD value per (p, q) pair."""
        scenarios = [
            Scenario(
                name="one p one q",
                given=Given(
                    data={
                        "data_p": [_make_signal()],
                        "data_q": [_make_signal()],
                    }
                ),
                expected=Expected(data={"element_count": 1}),
            ),
            Scenario(
                name="two p two q",
                given=Given(
                    data={
                        "data_p": [_make_signal(), _make_signal()],
                        "data_q": [_make_signal(), _make_signal()],
                    }
                ),
                expected=Expected(data={"element_count": 4}),
            ),
        ]

        for scenario in scenarios:
            with self.subTest(msg=scenario.name):
                mock_im.reset_mock()
                mock_im.kld.return_value = 0.5

                strategy = KLDStrategy(DiscreteEntropyConfig())
                data_input = KLDStrategyDataInput(
                    data_p=scenario.given.data["data_p"],
                    data_q=scenario.given.data["data_q"],
                )

                result = strategy.run_analysis(data_input)

                self.assertIsInstance(result, MatrixResult)
                self.assertEqual(result.algorithm, "kullback_leiber_divergence")
                self.assertEqual(
                    len(result.matrix), scenario.expected.data["element_count"]
                )
                self.assertEqual(
                    mock_im.kld.call_count, scenario.expected.data["element_count"]
                )

    @pytest.mark.unit
    @pytest.mark.strategy
    @patch("nexus.analysis.strategies.kld_strategy.im")
    @patch("nexus.analysis.estimators.entropy.DiscreteEntropyEstimator")
    def test_run_analysis_labels(
        self, _mock_estimator_class: MagicMock, mock_im: MagicMock
    ) -> None:
        """Test that signal names are used as labels, falling back to index when absent."""
        scenarios = [
            Scenario(
                name="named signals",
                given=Given(
                    data={
                        "data_p": _make_signal(name="dist_p"),
                        "data_q": _make_signal(name="dist_q"),
                    }
                ),
                expected=Expected(data={"row": "dist_p", "column": "dist_q"}),
            ),
            Scenario(
                name="unnamed signals",
                given=Given(
                    data={
                        "data_p": _make_signal(name=None),
                        "data_q": _make_signal(name=None),
                    }
                ),
                expected=Expected(data={"row": "data_p_0", "column": "data_q_0"}),
            ),
        ]

        for scenario in scenarios:
            with self.subTest(msg=scenario.name):
                mock_im.reset_mock()
                mock_im.kld.return_value = 0.5

                strategy = KLDStrategy(DiscreteEntropyConfig())
                data_input = KLDStrategyDataInput(
                    data_p=[scenario.given.data["data_p"]],
                    data_q=[scenario.given.data["data_q"]],
                )

                result = strategy.run_analysis(data_input)

                self.assertEqual(result.matrix[0].row, scenario.expected.data["row"])
                self.assertEqual(
                    result.matrix[0].column, scenario.expected.data["column"]
                )


class TestIntegrationKLDStrategy(unittest.TestCase):
    """Integration tests for KLDStrategy against real infomeasure."""

    @pytest.mark.integration
    @pytest.mark.strategy
    def test_kld_of_identical_distributions_is_zero(self) -> None:
        """KLD(P || P) = 0: divergence from a distribution to itself is zero."""
        data = np.tile([0.0, 1.0], 50)
        sig = AnalogSignal(data * pq.mV, sampling_rate=1.0 * pq.kHz)

        result = KLDStrategy(DiscreteEntropyConfig()).run_analysis(
            KLDStrategyDataInput(data_p=[sig], data_q=[sig])
        )

        self.assertAlmostEqual(result.matrix[0].value, 0.0, places=10)

    @pytest.mark.integration
    @pytest.mark.strategy
    def test_kld_of_different_distributions_is_positive(self) -> None:
        """KLD(P || Q) > 0 when P != Q (positivity of KL divergence)."""
        data_p = np.tile([0.0, 1.0], 50)
        data_q = np.tile([0.0, 0.0, 0.0, 1.0], 25)
        sig_p = AnalogSignal(data_p * pq.mV, sampling_rate=1.0 * pq.kHz)
        sig_q = AnalogSignal(data_q * pq.mV, sampling_rate=1.0 * pq.kHz)

        result = KLDStrategy(DiscreteEntropyConfig()).run_analysis(
            KLDStrategyDataInput(data_p=[sig_p], data_q=[sig_q])
        )

        self.assertGreater(result.matrix[0].value, 0.0)
