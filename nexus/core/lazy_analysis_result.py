"""Deferred analysis result that computes on first export."""

from nexus.analysis.interfaces import AnalysisResult, AnalysisStrategy
from nexus.core.interfaces import SignalProxy
from nexus.exportation.interfaces import ExportationStrategy


class LazyAnalysisResult:
    """Holds analysis inputs and defers computation until accept() is called.

    The actual strategy run is triggered the first time an ExportationStrategy
    visits this result via accept(). Subsequent calls reuse the cached
    AnalysisResult without re-running the strategy.

    Parameters
    ----------
    input_proxies : dict of {str: list of SignalProxy}
        Named groups of input proxies that will be loaded and passed to the
        analysis strategy when computation is triggered.
    analysis_strategy : AnalysisStrategy
        Strategy that computes the analysis measure over the loaded signals.
    """

    def __init__(
        self,
        input_proxies: dict[str, list[SignalProxy]],
        analysis_strategy: AnalysisStrategy,
    ):
        self._input_proxies = input_proxies
        self._analysis_strategy = analysis_strategy
        self._analysis_result: AnalysisResult | None = None

    def _compute(self) -> None:
        signals_dict = {
            key: [proxy.load() for proxy in proxies]
            for key, proxies in self._input_proxies.items()
        }
        input_data = self._analysis_strategy.data_input_type(**signals_dict)
        self._analysis_result = self._analysis_strategy.run_analysis(input_data)

    def accept(self, exporter_strategy: ExportationStrategy) -> None:
        """Compute the analysis if needed, then dispatch the result to the exporter.

        Parameters
        ----------
        exporter_strategy : ExportationStrategy
            Visitor that receives the AnalysisResult and writes or renders it.
        """
        if self._analysis_result is None:
            self._compute()
        self._analysis_result.accept(exporter_strategy)
