"""Analysis strategy for computing pairwise Kullback-Leibler divergence."""

from dataclasses import dataclass

from neo.core.dataobject import DataObject
import infomeasure as im

from nexus.analysis.estimators.entropy import EntropyConfig
from nexus.analysis.interfaces import EstimatorBasedAnalysisStrategy
from nexus.analysis.results.matrix_result import MatrixResult, MatrixElement
from nexus.common.interfaces import FilterCriteria
from nexus.types import Criteria


@dataclass
class KLDStrategyFilterCriteria(FilterCriteria):
    """Filter criteria for selecting signals used in KL divergence computation.

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
class KLDStrategyDataInput:
    """Data input container for KL divergence computation.

    Parameters
    ----------
    data_p : list[DataObject]
        Reference distribution signals (rows in the result matrix).
    data_q : list[DataObject]
        Approximating distribution signals (columns in the result matrix).
    """

    data_p: list[DataObject]
    data_q: list[DataObject]


class KLDStrategy(EstimatorBasedAnalysisStrategy):
    """Analysis strategy that computes pairwise Kullback-Leibler divergence D(P||Q).

    Produces a matrix result where entry (i, j) is D(data_p[i] || data_q[j]).

    Parameters
    ----------
    config : EntropyConfig
        Estimator configuration specifying the estimation approach.
    """

    _ALGORITHM_NAME = "kullback_leiber_divergence"

    def __init__(self, config: EntropyConfig) -> None:
        super().__init__(config)

    @property
    def filter_criteria_type(self) -> type[KLDStrategyFilterCriteria]:
        """Return the filter criteria type for KL divergence analysis.

        Returns
        -------
        type[KLDStrategyFilterCriteria]
            The criteria class used to select input signals.
        """
        return KLDStrategyFilterCriteria

    @property
    def data_input_type(self) -> type[KLDStrategyDataInput]:
        """Return the data input type for KL divergence analysis.

        Returns
        -------
        type[KLDStrategyDataInput]
            The data input class for this strategy.
        """
        return KLDStrategyDataInput

    def _get_estimator_result(
        self, signal_p: DataObject, signal_q: DataObject
    ) -> float:
        approach = self._config.get_approach_name()
        return im.kld(signal_p, signal_q, approach=approach, **self._estimator_kwargs)

    def run_analysis(self, data_input: KLDStrategyDataInput) -> MatrixResult:
        """Compute pairwise KL divergence for all (data_p, data_q) signal pairs.

        Parameters
        ----------
        data_input : KLDStrategyDataInput
            Loaded reference and approximating distribution signals.

        Returns
        -------
        MatrixResult
            A MatrixResult whose cells contain the KL divergence for each pair.
        """
        columns: list[str] = [
            signal_q.name or f"data_q_{j}"
            for j, signal_q in enumerate(data_input.data_q)
        ]
        matrix: list[MatrixElement] = []
        for i, signal_p in enumerate(data_input.data_p):
            row_label = signal_p.name or f"data_p_{i}"
            for signal_q, col_label in zip(data_input.data_q, columns):
                kld_result = self._get_estimator_result(signal_p, signal_q)
                matrix.append(
                    MatrixElement(row=row_label, column=col_label, value=kld_result)
                )
        return MatrixResult(
            algorithm=self._ALGORITHM_NAME,
            matrix=matrix,
            row_label="signal_p",
            col_label="signal_q",
        )
