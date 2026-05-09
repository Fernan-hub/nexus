from dataclasses import dataclass

from neo.core.dataobject import DataObject

from nexus.analysis.estimators.mutual_information import MutualInformationConfig
from nexus.analysis.interfaces import EstimatorBasedAnalysisStrategy
from nexus.analysis.results.matrix_result import MatrixResult, MatrixElement
from nexus.common.interfaces import FilterCriteria
from nexus.types import Criteria


@dataclass
class MutualInformationStrategyFilterCriteria(FilterCriteria):
    data_x: Criteria | None = None
    data_y: Criteria | None = None
    cond: Criteria | None = None


@dataclass
class MutualInformationStrategyDataInput:
    data_x: list[DataObject]
    data_y: list[DataObject]
    cond: list[DataObject] | None = None


class MutualInformationStrategy(EstimatorBasedAnalysisStrategy):
    _ALGORITHM_NAME = "mutual_information"

    def __init__(self, config: MutualInformationConfig) -> None:
        super().__init__(config)

    @property
    def filter_criteria_type(self) -> type[MutualInformationStrategyFilterCriteria]:
        return MutualInformationStrategyFilterCriteria

    @property
    def data_input_type(self) -> type[MutualInformationStrategyDataInput]:
        return MutualInformationStrategyDataInput

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
        self, data_input: MutualInformationStrategyDataInput
    ) -> MatrixResult:
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
