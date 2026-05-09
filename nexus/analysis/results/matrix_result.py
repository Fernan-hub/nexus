from dataclasses import dataclass, asdict
from functools import cached_property

import pandas as pd

from nexus.analysis.interfaces import AnalysisResult
from nexus.exportation.interfaces import ExportationStrategy


@dataclass
class MatrixElement:
    row: str
    column: str
    value: float

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


class MatrixResult(AnalysisResult):
    def __init__(
        self,
        algorithm: str,
        matrix: list[MatrixElement],
        row_label: str = "row",
        col_label: str = "column",
    ) -> None:
        super().__init__(algorithm)
        self.matrix = matrix
        self.row_label = row_label
        self.col_label = col_label

    @cached_property
    def matrix_df(self) -> pd.DataFrame:
        return pd.DataFrame([element.to_dict() for element in self.matrix])

    def accept(self, exporter_strategy: ExportationStrategy) -> None:
        exporter_strategy.export_matrix_result(self)
