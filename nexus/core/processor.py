"""Orchestrates processing strategies and registers their outputs into NeuroData."""

from typing import Any

from nexus.common.interfaces import FilterCriteria
from nexus.core.neuro_data import NeuroData
from nexus.core.operation_node import OperationNode
from nexus.core.proxies import ComputedSignalProxy
from nexus.processing.interfaces import ProcessingStrategy


class Processor:
    """Drives the processing phase of the Nexus pipeline.

    Translates a FilterCriteria into proxy lookups, delegates execution
    planning to the strategy, and registers all output ComputedSignalProxies
    back into NeuroData - without triggering any computation.

    Parameters
    ----------
    data : NeuroData
        The proxy registry from which input proxies are resolved and into which
        output proxies are registered.
    """

    def __init__(self, data: NeuroData) -> None:
        self._data = data

    def process(
        self,
        processing_strategy: ProcessingStrategy,
        filter_criteria: FilterCriteria,
        custom_annotations: dict[str, Any] | None = None,
    ) -> None:
        """Register output proxies for every node in the strategy's execution plan.

        custom_annotations are merged into every output proxy's annotation dict,
        letting callers tag outputs without modifying the strategy.

        Parameters
        ----------
        processing_strategy : ProcessingStrategy
            Strategy that defines the execution plan and the apply() computation.
        filter_criteria : FilterCriteria
            Criteria used to resolve the input proxies from NeuroData.
        custom_annotations : dict of {str: Any} or None, optional
            Extra annotations merged into each output proxy on top of those
            declared by the strategy's NodeDefinition. Defaults to no extras.
        """
        custom_annotations = (
            custom_annotations if custom_annotations is not None else {}
        )

        proxies_by_criteria = self._data.get_proxies_by_criteria_dict(
            filter_criteria.to_dict()
        )
        input_proxies = processing_strategy.proxy_input_type(**proxies_by_criteria)
        execution_plan = processing_strategy.infer_execution_plan(input_proxies)

        for node_def in execution_plan:
            operation_node = OperationNode(
                node_def.input_proxies_dict, processing_strategy
            )

            for index, annotations in enumerate(node_def.output_annotations):
                final_annotations = {**annotations, **custom_annotations}
                self._data.register_proxy(
                    ComputedSignalProxy(operation_node, index, final_annotations)
                )
