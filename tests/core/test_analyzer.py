"""Tests for Analyzer."""

import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import quantities as pq
from neo.core import AnalogSignal

from nexus.analysis.interfaces import AnalysisStrategy
from nexus.common.interfaces import FilterCriteria
from nexus.core.analyzer import Analyzer
from nexus.core.interfaces import SignalProxy
from nexus.core.lazy_analysis_result import LazyAnalysisResult
from nexus.core.neuro_data import NeuroData
from nexus.core.proxies import NeoSignalProxy


class TestAnalyzer(unittest.TestCase):
    """Unit tests for Analyzer."""

    @pytest.mark.unit
    @pytest.mark.core
    @patch("nexus.core.analyzer.LazyAnalysisResult")
    def test_analyze_data_returns_lazy_analysis_result(
        self, mock_lazy_result_cls: MagicMock
    ) -> None:
        """Returns a LazyAnalysisResult built from the proxies dict and strategy."""
        proxy = MagicMock(spec=SignalProxy)
        proxies_by_criteria = {"source": [proxy]}
        data = MagicMock(spec=NeuroData)
        data.get_proxies_by_criteria_dict.return_value = proxies_by_criteria

        strategy = MagicMock(spec=AnalysisStrategy)

        filter_criteria = MagicMock(spec=FilterCriteria)
        filter_criteria.to_dict.return_value = {"source": {"model": "efish"}}

        analyzer = Analyzer(data)
        result = analyzer.analyze_data(strategy, filter_criteria)

        mock_lazy_result_cls.assert_called_once_with(proxies_by_criteria, strategy)
        self.assertIs(result, mock_lazy_result_cls.return_value)

    @pytest.mark.unit
    @pytest.mark.core
    def test_analyze_data_queries_neuro_data_with_non_none_criteria(self) -> None:
        """Passes non-None criteria fields to get_proxies_by_criteria_dict."""
        data = MagicMock(spec=NeuroData)
        data.get_proxies_by_criteria_dict.return_value = {}

        strategy = MagicMock(spec=AnalysisStrategy)

        filter_criteria = MagicMock(spec=FilterCriteria)
        filter_criteria.to_dict.return_value = {"source": {"model": "efish"}}

        analyzer = Analyzer(data)
        analyzer.analyze_data(strategy, filter_criteria)

        data.get_proxies_by_criteria_dict.assert_called_once_with(
            {"source": {"model": "efish"}}
        )

    @pytest.mark.unit
    @pytest.mark.core
    def test_analyze_data_filters_none_criteria_values(self) -> None:
        """Strips criteria keys whose value is None before querying NeuroData."""
        data = MagicMock(spec=NeuroData)
        data.get_proxies_by_criteria_dict.return_value = {}

        strategy = MagicMock(spec=AnalysisStrategy)

        filter_criteria = MagicMock(spec=FilterCriteria)
        filter_criteria.to_dict.return_value = {
            "source": {"model": "efish"},
            "target": None,
        }

        analyzer = Analyzer(data)
        analyzer.analyze_data(strategy, filter_criteria)

        data.get_proxies_by_criteria_dict.assert_called_once_with(
            {"source": {"model": "efish"}}
        )


@dataclass
class _StubFilterCriteria(FilterCriteria):
    source: object = None


class _StubAnalysisStrategy(AnalysisStrategy):
    @property
    def filter_criteria_type(self):  # type: ignore[override]
        raise NotImplementedError

    @property
    def data_input_type(self):  # type: ignore[override]
        raise NotImplementedError

    def run_analysis(self, data_input):  # type: ignore[override]
        raise NotImplementedError


class TestIntegrationAnalyzer(unittest.TestCase):
    """Integration tests for Analyzer with real NeuroData."""

    def setUp(self) -> None:
        self.neuro_data = NeuroData()
        signal = AnalogSignal(
            np.array([[1.0]]) * pq.mV,
            sampling_rate=1.0 * pq.kHz,
            model="efish",
        )
        self.proxy = NeoSignalProxy(data_object=signal)
        self.neuro_data.register_proxy(self.proxy)

    @pytest.mark.integration
    @pytest.mark.core
    def test_analyze_data_captures_matching_proxies(self) -> None:
        """Result contains the proxies matching filter criteria in real NeuroData."""
        strategy = _StubAnalysisStrategy()
        filter_criteria = _StubFilterCriteria(source={"model": "efish"})

        analyzer = Analyzer(self.neuro_data)
        result = analyzer.analyze_data(strategy, filter_criteria)

        self.assertIsInstance(result, LazyAnalysisResult)
        self.assertIn(self.proxy, result._input_proxies["source"])
