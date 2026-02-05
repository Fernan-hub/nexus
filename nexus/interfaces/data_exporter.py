from nexus.models.results import AnalysisResult, ConnectivityMatrixResult


class DataExporter:
    def export_connectivity_matrix(self, result: ConnectivityMatrixResult) -> None:
        self.__raise_not_supported(result)
    
    def __raise_not_supported(self, result: AnalysisResult) -> None:
        raise NotImplementedError(
            f"Data exporter '{self.__class__.__name__}' does not support exporting "
            f"results of type '{result.__class__.__name__}'."
        )