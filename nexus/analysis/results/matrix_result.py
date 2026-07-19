"""Analysis result holding a pairwise matrix of values."""

from dataclasses import dataclass, asdict
from functools import cached_property

import pandas as pd

from nexus.analysis.interfaces import AnalysisResult
from nexus.exportation.interfaces import ExportationStrategy


@dataclass
class MatrixElement:
    """A single cell in a pairwise analysis matrix.

    Parameters
    ----------
    row : str
        Label identifying the row (source) signal.
    column : str
        Label identifying the column (target) signal.
    value : float
        The computed pairwise measure value.
    """

    row: str
    column: str
    value: float

    def to_dict(self) -> dict[str, str | float]:
        """Return the element as a plain dictionary.

        Returns
        -------
        dict[str, str or float]
            Dictionary with keys ``'row'``, ``'column'``, and ``'value'``.
        """
        return asdict(self)


class MatrixResult(AnalysisResult):
    """Analysis result that wraps a pairwise matrix of signal measurements.

    Parameters
    ----------
    algorithm : str
        Name of the analysis algorithm that produced this result.
    matrix : list[MatrixElement]
        Flat list of pairwise elements representing the matrix cells.
    row_label : str, optional
        Display name for the row axis, by default ``'row'``.
    col_label : str, optional
        Display name for the column axis, by default ``'column'``.
    """

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
        """Return the matrix as a tidy DataFrame with one row per element.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``'row'``, ``'column'``, and ``'value'``.
        """
        return pd.DataFrame([element.to_dict() for element in self.matrix])

    def accept(self, exporter_strategy: ExportationStrategy) -> None:
        """Dispatch this result to the exporter's matrix result handler.

        Parameters
        ----------
        exporter_strategy : ExportationStrategy
            The exporter that will consume this result.
        """
        exporter_strategy.export_matrix_result(self)
