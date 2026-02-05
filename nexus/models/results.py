from abc import ABC, abstractmethod
import pandas as pd

from nexus.interfaces.data_exporter import DataExporter


class AnalysisResult(ABC):
    
    @abstractmethod
    def accept(self, exporter: DataExporter) -> None:
        pass


class ConnectivityMatrixResult(AnalysisResult):
    matrix: pd.DataFrame
    
    def accept(self, exporter: DataExporter) -> None:
        exporter.export_connectivity_matrix(self)
