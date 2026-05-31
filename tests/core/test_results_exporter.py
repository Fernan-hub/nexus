"""Tests for ResultsExporter."""

import unittest
from unittest.mock import MagicMock

import pytest

from nexus.core.lazy_analysis_result import LazyAnalysisResult
from nexus.core.results_exporter import ResultsExporter
from nexus.exportation.interfaces import ExportationStrategy


class TestResultsExporter(unittest.TestCase):
    """Unit tests for ResultsExporter."""

    @pytest.mark.unit
    @pytest.mark.core
    def test_export_result(self) -> None:
        lazy_result = MagicMock(spec=LazyAnalysisResult)
        exporter_strategy = MagicMock(spec=ExportationStrategy)

        results_exporter = ResultsExporter(lazy_result)
        results_exporter.export_result(exporter_strategy)

        lazy_result.accept.assert_called_once_with(exporter_strategy)
