"""Tests for LazyAnalysisResult."""

import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest
import quantities as pq
from neo.core import AnalogSignal

from nexus.analysis.interfaces import AnalysisResult, AnalysisStrategy
from nexus.core.interfaces import SignalProxy
from nexus.core.lazy_analysis_result import LazyAnalysisResult
from nexus.core.proxies import NeoSignalProxy
from nexus.exportation.interfaces import ExportationStrategy
from nexus.exportation.models import ExportationStrategyConfig


class TestLazyAnalysisResult(unittest.TestCase):
    """Unit tests for LazyAnalysisResult."""

    def setUp(self) -> None:
        self.proxy = MagicMock(spec=SignalProxy)
        self.proxy.load.return_value = MagicMock()
        self.analysis_result = MagicMock(spec=AnalysisResult)
        self.exporter = MagicMock(spec=ExportationStrategy)
        self.data_input_sentinel = MagicMock(name="data_input")
        self.strategy = MagicMock(spec=AnalysisStrategy)
        self.strategy.data_input_type.return_value = self.data_input_sentinel
        self.strategy.run_analysis.return_value = self.analysis_result

    @pytest.mark.unit
    @pytest.mark.core
    def test_accept(self) -> None:
        """Loads proxies, runs analysis, and dispatches the result to the exporter."""
        lazy_result = LazyAnalysisResult(
            input_proxies={"source": [self.proxy]},
            analysis_strategy=self.strategy,
        )
        lazy_result.accept(self.exporter)

        self.proxy.load.assert_called_once_with()
        self.strategy.run_analysis.assert_called_once_with(self.data_input_sentinel)
        self.analysis_result.accept.assert_called_once_with(self.exporter)

    @pytest.mark.unit
    @pytest.mark.core
    def test_accept_does_not_recompute_on_second_call(self) -> None:
        """Calls proxy.load and run_analysis once even when accept is called twice."""
        lazy_result = LazyAnalysisResult(
            input_proxies={"source": [self.proxy]},
            analysis_strategy=self.strategy,
        )
        lazy_result.accept(self.exporter)
        lazy_result.accept(self.exporter)

        self.proxy.load.assert_called_once_with()
        self.strategy.run_analysis.assert_called_once_with(self.data_input_sentinel)

    @pytest.mark.unit
    @pytest.mark.core
    def test_accept_builds_data_input_from_multi_group_proxies(self) -> None:
        """Passes signals by group name to data_input_type with multiple groups."""
        proxy_src = MagicMock(spec=SignalProxy)
        signal_src = MagicMock()
        proxy_src.load.return_value = signal_src

        proxy_tgt = MagicMock(spec=SignalProxy)
        signal_tgt = MagicMock()
        proxy_tgt.load.return_value = signal_tgt

        lazy_result = LazyAnalysisResult(
            input_proxies={"source": [proxy_src], "target": [proxy_tgt]},
            analysis_strategy=self.strategy,
        )
        lazy_result.accept(self.exporter)

        self.strategy.data_input_type.assert_called_once_with(
            source=[signal_src], target=[signal_tgt]
        )


@dataclass
class _StubDataInput:
    signals: list


class _StubAnalysisResult(AnalysisResult):
    def __init__(self, received_signals: list) -> None:
        super().__init__(algorithm="stub")
        self.received_signals = received_signals

    def accept(self, exporter_strategy: ExportationStrategy) -> None:
        exporter_strategy.export_scalar_result(self)


class _StubAnalysisStrategy(AnalysisStrategy):
    def __init__(self) -> None:
        self.last_result: _StubAnalysisResult | None = None

    @property
    def filter_criteria_type(self):  # type: ignore[override]
        raise NotImplementedError

    @property
    def data_input_type(self):  # type: ignore[override]
        return _StubDataInput

    def run_analysis(self, data_input: _StubDataInput) -> _StubAnalysisResult:
        self.last_result = _StubAnalysisResult(data_input.signals)
        return self.last_result


class _StubExportationStrategy(ExportationStrategy):
    def __init__(self) -> None:
        super().__init__(ExportationStrategyConfig())
        self.received_result: _StubAnalysisResult | None = None

    def export_scalar_result(self, scalar_result: _StubAnalysisResult) -> None:
        self.received_result = scalar_result


class TestIntegrationLazyAnalysisResult(unittest.TestCase):
    """Integration tests for LazyAnalysisResult with real proxies and strategy."""

    def setUp(self) -> None:
        self.signal = AnalogSignal(
            np.array([[1.0], [2.0]]) * pq.mV,
            sampling_rate=1.0 * pq.kHz,
        )
        self.proxy = NeoSignalProxy(data_object=self.signal)

    @pytest.mark.integration
    @pytest.mark.core
    def test_accept(self) -> None:
        """End-to-end: proxy loads, analysis runs, result reaches the exporter."""
        strategy = _StubAnalysisStrategy()
        exporter = _StubExportationStrategy()

        lazy_result = LazyAnalysisResult(
            input_proxies={"signals": [self.proxy]},
            analysis_strategy=strategy,
        )
        lazy_result.accept(exporter)

        self.assertIs(strategy.last_result.received_signals[0], self.signal)
        self.assertIs(exporter.received_result, strategy.last_result)
