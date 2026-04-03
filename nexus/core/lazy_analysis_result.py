from nexus.analysis.interfaces import AnalysisResult, AnalysisStrategy
from nexus.core.interfaces import SignalProxy
from nexus.exporting.interfaces import ExporterStrategy


class LazyAnalysisResult:
    def __init__(
        self, input_proxies: list[SignalProxy], analysis_strategy: AnalysisStrategy
    ):
        self._input_proxies = input_proxies
        self._analysis_strategy = analysis_strategy
        self._analysis_result: AnalysisResult | None = None

    def compute(self) -> AnalysisResult:
        if self._analysis_result is None:
            signals = [proxy.load() for proxy in self._input_proxies]
            self._analysis_result = self._analysis_strategy.run_analysis(signals)
        return self._analysis_result

    def accept(self, exporter_strategy: ExporterStrategy) -> None:
        self.compute()
        self._analysis_result.accept(exporter_strategy)
