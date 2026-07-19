"""Analysis strategy for computing pairwise (conditional) transfer entropy."""

from dataclasses import dataclass

from neo.core.dataobject import DataObject

from nexus.analysis.estimators.transfer_entropy import TransferEntropyConfig
from nexus.analysis.interfaces import EstimatorBasedAnalysisStrategy
from nexus.analysis.results.matrix_result import MatrixResult, MatrixElement
from nexus.common.interfaces import FilterCriteria
from nexus.types import Criteria


@dataclass
class TransferEntropyStrategyFilterCriteria(FilterCriteria):
    """Filter criteria for selecting signals used in transfer entropy computation.

    Parameters
    ----------
    sources : Criteria or None, optional
        Criteria for selecting source signals (rows in the result matrix).
    dests : Criteria or None, optional
        Criteria for selecting destination signals (columns in the result matrix).
    cond : Criteria or None, optional
        Criteria for selecting the conditioning signal. When provided,
        conditional transfer entropy TE(X->Y|Z) is computed.
    """

    sources: Criteria | None = None
    dests: Criteria | None = None
    cond: Criteria | None = None


@dataclass
class TransferEntropyStrategyDataInput:
    """Data input container for transfer entropy computation.

    Parameters
    ----------
    sources : list[DataObject]
        Source signals (rows in the result matrix).
    dests : list[DataObject]
        Destination signals (columns in the result matrix).
    cond : list[DataObject] or None, optional
        Conditioning signal. When provided, conditional TE is computed.
    """

    sources: list[DataObject]
    dests: list[DataObject]
    cond: list[DataObject] | None = None


class TransferEntropyStrategy(EstimatorBasedAnalysisStrategy):
    """Analysis strategy that computes pairwise (conditional) transfer entropy.

    Produces a matrix result where entry (i, j) is TE(sources[i] -> dests[j]).

    Parameters
    ----------
    config : TransferEntropyConfig
        Estimator configuration specifying the estimation approach.
    """

    _ALGORITHM_NAME = "transfer_entropy"

    def __init__(self, config: TransferEntropyConfig) -> None:
        super().__init__(config)

    @property
    def filter_criteria_type(self) -> type[TransferEntropyStrategyFilterCriteria]:
        """Return the filter criteria type for transfer entropy analysis.

        Returns
        -------
        type[TransferEntropyStrategyFilterCriteria]
            The criteria class used to select input signals.
        """
        return TransferEntropyStrategyFilterCriteria

    @property
    def data_input_type(self) -> type[TransferEntropyStrategyDataInput]:
        """Return the data input type for transfer entropy analysis.

        Returns
        -------
        type[TransferEntropyStrategyDataInput]
            The data input class for this strategy.
        """
        return TransferEntropyStrategyDataInput

    def _get_estimator_result(
        self, signal_x: DataObject, signal_y: DataObject, cond: list[DataObject] | None
    ) -> float:
        if cond is None or len(cond) == 0:
            estimator = self._estimator_class(
                signal_x, signal_y, **self._estimator_kwargs
            )
        else:
            estimator_class = self._config.get_conditional_estimator_class()
            estimator = estimator_class(
                signal_x, signal_y, cond=cond[0], **self._estimator_kwargs
            )
        return estimator.result()

    def run_analysis(
        self, data_input: TransferEntropyStrategyDataInput
    ) -> MatrixResult:
        """Compute pairwise transfer entropy for all (source, destination) signal pairs.

        Parameters
        ----------
        data_input : TransferEntropyStrategyDataInput
            Loaded source, destination, and optional conditioning signals.

        Returns
        -------
        MatrixResult
            A MatrixResult whose cells contain the TE value for each signal pair.
        """
        columns: list[str] = [
            dst.name or f"dest_{j}" for j, dst in enumerate(data_input.dests)
        ]
        matrix: list[MatrixElement] = []
        for i, src in enumerate(data_input.sources):
            row_label = src.name or f"source_{i}"
            for dst, col_label in zip(data_input.dests, columns):
                te_result = self._get_estimator_result(src, dst, data_input.cond)
                matrix.append(
                    MatrixElement(row=row_label, column=col_label, value=te_result)
                )
        return MatrixResult(
            algorithm=self._ALGORITHM_NAME,
            matrix=matrix,
            row_label="source",
            col_label="destination",
        )
