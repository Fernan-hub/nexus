from dataclasses import dataclass

from neo.core.dataobject import DataObject

from nexus.analysis.estimators.transfer_entropy import TransferEntropyConfig
from nexus.analysis.interfaces import EstimatorBasedAnalysisStrategy
from nexus.analysis.results.matrix_result import MatrixResult, MatrixElement
from nexus.common.interfaces import FilterCriteria
from nexus.types import Criteria


@dataclass
class TransferEntropyStrategyFilterCriteria(FilterCriteria):
    sources: Criteria | None = None
    dests: Criteria | None = None
    cond: Criteria | None = None


@dataclass
class TransferEntropyStrategyDataInput:
    sources: list[DataObject]
    dests: list[DataObject]
    cond: list[DataObject] | None = None


class TransferEntropyStrategy(EstimatorBasedAnalysisStrategy):
    _ALGORITHM_NAME = "transfer_entropy"

    def __init__(self, config: TransferEntropyConfig) -> None:
        super().__init__(config)

    @property
    def filter_criteria_type(self) -> type[TransferEntropyStrategyFilterCriteria]:
        return TransferEntropyStrategyFilterCriteria

    @property
    def data_input_type(self) -> type[TransferEntropyStrategyDataInput]:
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
        return MatrixResult(algorithm=self._ALGORITHM_NAME, matrix=matrix)
