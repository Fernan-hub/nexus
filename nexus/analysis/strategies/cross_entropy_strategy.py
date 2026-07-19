"""Analysis strategy for computing pairwise cross-entropy between distributions."""

from dataclasses import dataclass

from neo.core.dataobject import DataObject

from nexus.analysis.estimators.entropy import EntropyConfig
from nexus.analysis.interfaces import EstimatorBasedAnalysisStrategy
from nexus.analysis.results.matrix_result import MatrixResult, MatrixElement
from nexus.common.interfaces import FilterCriteria
from nexus.types import Criteria


@dataclass
class CrossEntropyStrategyFilterCriteria(FilterCriteria):
    """Filter criteria for selecting signals used in cross-entropy computation.

    Parameters
    ----------
    data_p : Criteria or None, optional
        Criteria for selecting the reference distribution signals (rows).
    data_q : Criteria or None, optional
        Criteria for selecting the approximating distribution signals (columns).
    """

    data_p: Criteria | None = None
    data_q: Criteria | None = None


@dataclass
class CrossEntropyStrategyDataInput:
    """Data input container for cross-entropy computation.

    Parameters
    ----------
    data_p : list[DataObject]
        Reference distribution signals (rows in the result matrix).
    data_q : list[DataObject]
        Approximating distribution signals (columns in the result matrix).
    """

    data_p: list[DataObject]
    data_q: list[DataObject]


class CrossEntropyStrategy(EstimatorBasedAnalysisStrategy):
    """Analysis strategy that computes pairwise cross-entropy H(P, Q).

    Produces a matrix result where entry (i, j) is H(data_p[i], data_q[j]).

    Parameters
    ----------
    config : EntropyConfig
        Estimator configuration specifying the estimation approach.
    """

    _ALGORITHM_NAME = "cross_entropy"

    def __init__(self, config: EntropyConfig) -> None:
        super().__init__(config)

    @property
    def filter_criteria_type(self) -> type[CrossEntropyStrategyFilterCriteria]:
        """Return the filter criteria type for cross-entropy analysis.

        Returns
        -------
        type[CrossEntropyStrategyFilterCriteria]
            The criteria class used to select input signals.
        """
        return CrossEntropyStrategyFilterCriteria

    @property
    def data_input_type(self) -> type[CrossEntropyStrategyDataInput]:
        """Return the data input type for cross-entropy analysis.

        Returns
        -------
        type[CrossEntropyStrategyDataInput]
            The data input class for this strategy.
        """
        return CrossEntropyStrategyDataInput

    def _get_estimator_result(
        self, signal_p: DataObject, signal_q: DataObject
    ) -> float:
        estimator = self._estimator_class(signal_p, signal_q, **self._estimator_kwargs)
        return estimator.result()

    def run_analysis(self, data_input: CrossEntropyStrategyDataInput) -> MatrixResult:
        """Compute pairwise cross-entropy for all (data_p, data_q) signal pairs.

        Parameters
        ----------
        data_input : CrossEntropyStrategyDataInput
            Loaded reference and approximating distribution signals.

        Returns
        -------
        MatrixResult
            A MatrixResult whose cells contain the cross-entropy value for each pair.
        """
        columns: list[str] = [
            signal_q.name or f"data_q_{j}"
            for j, signal_q in enumerate(data_input.data_q)
        ]
        matrix: list[MatrixElement] = []
        for i, signal_p in enumerate(data_input.data_p):
            row_label = signal_p.name or f"data_p_{i}"
            for signal_q, col_label in zip(data_input.data_q, columns):
                cross_entropy_result = self._get_estimator_result(signal_p, signal_q)
                matrix.append(
                    MatrixElement(
                        row=row_label, column=col_label, value=cross_entropy_result
                    )
                )
        return MatrixResult(
            algorithm=self._ALGORITHM_NAME,
            matrix=matrix,
            row_label="signal_p",
            col_label="signal_q",
        )
