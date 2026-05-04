from dataclasses import dataclass

from neo.core.dataobject import DataObject
import pandas as pd

from nexus.analysis.estimators.entropy import EntropyConfig
from nexus.analysis.interfaces import EstimatorBasedAnalysisStrategy
from nexus.common.interfaces import FilterCriteria
from nexus.types import Criteria
from nexus.analysis.results.vector_result import VectorResult


@dataclass
class EntropyStrategyFilterCriteria(FilterCriteria):
    data: Criteria | None = None
    cond: Criteria | None = None


@dataclass
class EntropyStrategyDataInput:
    data: list[DataObject]
    cond: list[DataObject] | None = None


class EntropyStrategy(EstimatorBasedAnalysisStrategy):
    _ALGORITHM_NAME = "entropy"

    def __init__(self, config: EntropyConfig) -> None:
        super().__init__(config)

    @property
    def filter_criteria_type(self) -> type[EntropyStrategyFilterCriteria]:
        return EntropyStrategyFilterCriteria

    @property
    def data_input_type(self) -> type[EntropyStrategyDataInput]:
        return EntropyStrategyDataInput

    def _compute_joint_entropy(
        self, signal: DataObject, cond: list[DataObject]
    ) -> float:
        joint_data = tuple(signal, cond[0])
        estimator = self._estimator_class(joint_data, **self._estimator_kwargs)
        return estimator.result()

    def _compute_simple_entropy(self, signal: DataObject) -> float:
        estimator = self._estimator_class(signal, **self._estimator_kwargs)
        return estimator.result()

    def _get_estimator_result(
        self, signal: DataObject, cond: list[DataObject] | None
    ) -> float:
        if cond is None or len(cond) == 0:
            return self._compute_simple_entropy(signal)

        joint_entropy = self._compute_joint_entropy(signal, cond)
        cond_entropy = self._compute_simple_entropy(cond[0])
        return joint_entropy - cond_entropy

    def _get_column_name(self, cond: list[DataObject] | None) -> str:
        if cond is None or len(cond) == 0:
            return "entropy"
        if cond[0].name is not None:
            return f"cond_entropy_given_{cond[0].name}"
        return "cond_entropy"

    def run_analysis(self, data_input: EntropyStrategyDataInput) -> VectorResult:
        results_list = [
            self._get_estimator_result(data, data_input.cond)
            for data in data_input.data
        ]
        columns = [self._get_column_name(data_input.cond)]
        results_df = pd.DataFrame(results_list, columns=columns)
        return VectorResult(algorithm=self._ALGORITHM_NAME, vector=results_df)
