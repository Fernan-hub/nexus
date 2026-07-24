"""Builds lazy analysis results from NeuroData proxies and analysis strategies."""

from nexus.analysis.interfaces import AnalysisStrategy
from nexus.common.interfaces import FilterCriteria
from nexus.core.lazy_analysis_result import LazyAnalysisResult
from nexus.core.neuro_data import NeuroData


class Analyzer:
    """Drives the analysis phase of the Nexus pipeline.

    Resolves proxies from NeuroData using the provided criteria, then wraps
    them with the chosen strategy in a LazyAnalysisResult - no computation
    happens until an exporter calls accept() on the result.

    Parameters
    ----------
    data : NeuroData
        The proxy registry from which input proxies are resolved.
    """

    def __init__(self, data: NeuroData) -> None:
        self._data = data

    def analyze_data(
        self, analysis_strategy: AnalysisStrategy, filter_criteria: FilterCriteria
    ) -> LazyAnalysisResult:
        """Return a LazyAnalysisResult wrapping the matched proxies and strategy.

        Parameters
        ----------
        analysis_strategy : AnalysisStrategy
            Strategy that will compute the analysis measure when the result is
            exported.
        filter_criteria : FilterCriteria
            Criteria used to resolve the input proxies from NeuroData. Entries
            with None values are excluded from the lookup.

        Returns
        -------
        LazyAnalysisResult
            A deferred result that computes and dispatches to an exporter on
            the first call to accept().
        """
        criteria_dict = {
            k: v for k, v in filter_criteria.to_dict().items() if v is not None
        }
        input_proxies = self._data.get_proxies_by_criteria_dict(criteria_dict)
        return LazyAnalysisResult(input_proxies, analysis_strategy)
