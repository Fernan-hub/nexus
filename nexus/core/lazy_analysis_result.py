from nexus.analysis.interfaces import AnalysisResult, AnalysisStrategy
from nexus.core.interfaces import SignalProxy
from nexus.exportation.interfaces import ExportationStrategy


class LazyAnalysisResult:
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
        if self._analysis_result is None:
            self._compute()
        self._analysis_result.accept(exporter_strategy)
