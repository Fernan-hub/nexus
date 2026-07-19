"""Analysis strategy for computing pairwise (conditional) mutual information."""

from dataclasses import dataclass

import numpy as np
from neo.core.dataobject import DataObject

from nexus.analysis.estimators.mutual_information import MutualInformationConfig
from nexus.analysis.interfaces import EstimatorBasedAnalysisStrategy
from nexus.analysis.results.matrix_result import MatrixResult, MatrixElement
from nexus.common.interfaces import FilterCriteria
from nexus.types import Criteria


@dataclass
class MutualInformationStrategyFilterCriteria(FilterCriteria):
    """Filter criteria for selecting signals used in mutual information computation.

    Parameters
    ----------
    data_x : Criteria or None, optional
        Criteria for selecting the first set of signals (rows).
    data_y : Criteria or None, optional
        Criteria for selecting the second set of signals (columns).
    cond : Criteria or None, optional
        Criteria for selecting the conditioning signal. When provided,
        conditional mutual information I(X;Y|Z) is computed.
    """

    data_x: Criteria | None = None
    data_y: Criteria | None = None
    cond: Criteria | None = None


@dataclass
class MutualInformationStrategyDataInput:
    """Data input container for mutual information computation.

    Parameters
    ----------
    data_x : list[DataObject]
        First set of signals (rows in the result matrix).
    data_y : list[DataObject]
        Second set of signals (columns in the result matrix).
    cond : list[DataObject] or None, optional
        Conditioning signal. When provided, conditional MI is computed.
    """

    data_x: list[DataObject]
    data_y: list[DataObject]
    cond: list[DataObject] | None = None


class MutualInformationStrategy(EstimatorBasedAnalysisStrategy):
    """Analysis strategy that computes pairwise (conditional) mutual information.

    Produces a matrix result where entry (i, j) is I(data_x[i]; data_y[j]).

    Parameters
    ----------
    config : MutualInformationConfig
        Estimator configuration specifying the estimation approach.
    """

    _ALGORITHM_NAME = "mutual_information"

    def __init__(self, config: MutualInformationConfig) -> None:
        super().__init__(config)

    @property
    def filter_criteria_type(self) -> type[MutualInformationStrategyFilterCriteria]:
        """Return the filter criteria type for mutual information analysis.

        Returns
        -------
        type[MutualInformationStrategyFilterCriteria]
            The criteria class used to select input signals.
        """
        return MutualInformationStrategyFilterCriteria

    @property
    def data_input_type(self) -> type[MutualInformationStrategyDataInput]:
        """Return the data input type for mutual information analysis.

        Returns
        -------
        type[MutualInformationStrategyDataInput]
            The data input class for this strategy.
        """
        return MutualInformationStrategyDataInput

    def _get_estimator_result(
        self, signal_x: DataObject, signal_y: DataObject, cond: list[DataObject] | None
    ) -> float:
        x = np.squeeze(np.asarray(signal_x))
        y = np.squeeze(np.asarray(signal_y))
        if cond is None or len(cond) == 0:
            estimator = self._estimator_class(x, y, **self._estimator_kwargs)
        else:
            estimator_class = self._config.get_conditional_estimator_class()
            cond_squeezed = np.squeeze(np.asarray(cond[0]))
            estimator = estimator_class(
                x, y, cond=cond_squeezed, **self._estimator_kwargs
            )
        return estimator.result()

    def run_analysis(
        self, data_input: MutualInformationStrategyDataInput
    ) -> MatrixResult:
        """Compute pairwise mutual information for all (data_x, data_y) signal pairs.

        Parameters
        ----------
        data_input : MutualInformationStrategyDataInput
            Loaded signals for both axes and optional conditioning signal.

        Returns
        -------
        MatrixResult
            A MatrixResult whose cells contain the MI value for each signal pair.
        """
        columns: list[str] = [
            signal_y.name or f"data_y_{j}"
            for j, signal_y in enumerate(data_input.data_y)
        ]
        matrix: list[MatrixElement] = []
        for i, signal_x in enumerate(data_input.data_x):
            row_label = signal_x.name or f"data_x_{i}"
            for signal_y, col_label in zip(data_input.data_y, columns):
                mi_result = self._get_estimator_result(
                    signal_x, signal_y, data_input.cond
                )
                matrix.append(
                    MatrixElement(row=row_label, column=col_label, value=mi_result)
                )
        return MatrixResult(
            algorithm=self._ALGORITHM_NAME,
            matrix=matrix,
            row_label="signal_x",
            col_label="signal_y",
        )
