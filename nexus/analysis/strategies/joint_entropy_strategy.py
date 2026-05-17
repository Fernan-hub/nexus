from dataclasses import dataclass

from neo.core.dataobject import DataObject

from nexus.analysis.estimators.entropy import EntropyConfig
from nexus.analysis.interfaces import EstimatorBasedAnalysisStrategy
from nexus.common.interfaces import FilterCriteria
from nexus.types import Criteria
from nexus.analysis.results.scalar_result import ScalarResult


@dataclass
class JointEntropyStrategyFilterCriteria(FilterCriteria):
    data: Criteria | None = None


@dataclass
class JointEntropyStrategyDataInput:
    data: list[DataObject]


class JointEntropyStrategy(EstimatorBasedAnalysisStrategy):
    _ALGORITHM_NAME = "joint_entropy"

    def __init__(self, config: EntropyConfig) -> None:
        super().__init__(config)

    @property
    def filter_criteria_type(self) -> type[JointEntropyStrategyFilterCriteria]:
        return JointEntropyStrategyFilterCriteria

    @property
    def data_input_type(self) -> type[JointEntropyStrategyDataInput]:
        return JointEntropyStrategyDataInput

    def _get_estimator_result(self, signals: list[DataObject]) -> float:
        joint_data = tuple(signals)
        estimator = self._estimator_class(joint_data, **self._estimator_kwargs)
        return estimator.result()

    def _get_column_name(self, signals: list[DataObject] | None) -> str:
        signals_names = [
            signal.name or f"signal_{i}" for i, signal in enumerate(signals)
        ]
        return "Joint entropy of " + ", ".join(signals_names)

    def run_analysis(self, data_input: JointEntropyStrategyDataInput) -> ScalarResult:
        joint_entropy_result = self._get_estimator_result(data_input.data)
        return ScalarResult(
            algorithm=self._ALGORITHM_NAME,
            value=joint_entropy_result,
            label=self._get_column_name(data_input.data),
        )
