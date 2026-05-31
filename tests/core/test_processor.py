"""Tests for Processor."""

import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import quantities as pq
from neo.core import AnalogSignal
from neo.core.dataobject import DataObject

from nexus.common.interfaces import FilterCriteria
from nexus.core.interfaces import SignalProxy
from nexus.core.neuro_data import NeuroData
from nexus.core.processor import Processor
from nexus.core.proxies import ComputedSignalProxy, NeoSignalProxy
from nexus.models import NodeDefinition
from nexus.processing.interfaces import ProcessingStrategy
from tests.utils import Expected, Given, Scenario


class TestProcessor(unittest.TestCase):
    """Unit tests for Processor."""

    def setUp(self) -> None:
        self.mock_proxy = MagicMock(spec=SignalProxy)
        self.data = MagicMock(spec=NeuroData)
        self.data.get_proxies_by_criteria_dict.return_value = {
            "inputs": [self.mock_proxy]
        }
        self.strategy = MagicMock(spec=ProcessingStrategy)
        self.strategy.proxy_input_type = MagicMock()
        self.filter_criteria = MagicMock(spec=FilterCriteria)
        self.filter_criteria.to_dict.return_value = {"inputs": {"model": "efish"}}
        self.processor = Processor(self.data)

    @pytest.mark.unit
    @pytest.mark.core
    @patch("nexus.core.processor.ComputedSignalProxy")
    @patch("nexus.core.processor.OperationNode")
    def test_process(
        self, MockOperationNode: MagicMock, MockComputedSignalProxy: MagicMock
    ) -> None:
        """Creates OperationNode and proxy per node def, merging custom_annotations."""
        scenarios = [
            Scenario(
                name="no_custom_annotations",
                given=Given(
                    data={
                        "output_annotations": [{"model": "efish"}],
                        "custom_annotations": None,
                    }
                ),
                expected=Expected(
                    data={
                        "proxy_annotations": [{"model": "efish"}],
                    }
                ),
            ),
            Scenario(
                name="non_conflicting_custom_annotations",
                given=Given(
                    data={
                        "output_annotations": [{"channel": "A"}, {"channel": "B"}],
                        "custom_annotations": {"run": 1},
                    }
                ),
                expected=Expected(
                    data={
                        "proxy_annotations": [
                            {"channel": "A", "run": 1},
                            {"channel": "B", "run": 1},
                        ],
                    }
                ),
            ),
            Scenario(
                name="overriding_custom_annotations",
                given=Given(
                    data={
                        "output_annotations": [{"model": "efish"}],
                        "custom_annotations": {"model": "other"},
                    }
                ),
                expected=Expected(
                    data={
                        "proxy_annotations": [{"model": "other"}],
                    }
                ),
            ),
        ]

        for scenario in scenarios:
            with self.subTest(scenario.name):
                MockOperationNode.reset_mock()
                MockComputedSignalProxy.reset_mock()
                self.data.reset_mock()
                self.data.get_proxies_by_criteria_dict.return_value = {
                    "inputs": [self.mock_proxy]
                }

                node_def = NodeDefinition(
                    input_proxies_dict={"inputs": [self.mock_proxy]},
                    output_annotations=scenario.given.data["output_annotations"],
                )
                self.strategy.infer_execution_plan.return_value = [node_def]

                self.processor.process(
                    self.strategy,
                    self.filter_criteria,
                    custom_annotations=scenario.given.data["custom_annotations"],
                )

                expected_annotations = scenario.expected.data["proxy_annotations"]

                self.data.get_proxies_by_criteria_dict.assert_called_once_with(
                    {"inputs": {"model": "efish"}}
                )
                MockOperationNode.assert_called_once_with(
                    {"inputs": [self.mock_proxy]}, self.strategy
                )
                self.assertEqual(
                    MockComputedSignalProxy.call_count, len(expected_annotations)
                )
                for i, c in enumerate(MockComputedSignalProxy.call_args_list):
                    self.assertIs(c.args[0], MockOperationNode.return_value)
                    self.assertEqual(c.args[1], i)
                    self.assertEqual(c.args[2], expected_annotations[i])
                self.assertEqual(
                    self.data.register_proxy.call_count, len(expected_annotations)
                )
                for c in self.data.register_proxy.call_args_list:
                    self.assertIs(c.args[0], MockComputedSignalProxy.return_value)


@dataclass
class _StubFilterCriteria(FilterCriteria):
    inputs: object = None


@dataclass
class _StubProxyInput:
    inputs: list


@dataclass
class _StubDataInput:
    inputs: list


class _PassthroughStrategy(ProcessingStrategy):
    """Minimal pass-through strategy for integration testing."""

    supported_data_object_types = [AnalogSignal]

    @property
    def filter_criteria_type(self):  # type: ignore[override]
        return _StubFilterCriteria

    @property
    def proxy_input_type(self):  # type: ignore[override]
        return _StubProxyInput

    @property
    def data_input_type(self):  # type: ignore[override]
        return _StubDataInput

    def infer_execution_plan(self, input_proxies: _StubProxyInput) -> list:
        return [
            NodeDefinition(
                input_proxies_dict={"inputs": [proxy]},
                output_annotations=[{**proxy.annotations, "processed": True}],
            )
            for proxy in input_proxies.inputs
        ]

    def apply(self, input_data: _StubDataInput) -> list[DataObject]:
        return list(input_data.inputs)


class TestIntegrationProcessor(unittest.TestCase):
    """Integration tests for Processor with real NeuroData and proxies."""

    def setUp(self) -> None:
        self.signal = AnalogSignal(
            np.array([[1.0], [2.0]]) * pq.mV,
            sampling_rate=1.0 * pq.kHz,
            model="efish",
        )
        self.strategy = _PassthroughStrategy()

    @pytest.mark.integration
    @pytest.mark.core
    def test_process(self) -> None:
        """Registers a ComputedSignalProxy in NeuroData with merged annotations."""
        scenarios = [
            Scenario(
                name="no_custom_annotations",
                given=Given(data={"custom_annotations": None}),
                expected=Expected(
                    data={
                        "annotations": {"model": "efish", "processed": True},
                    }
                ),
            ),
            Scenario(
                name="non_conflicting_custom_annotations",
                given=Given(data={"custom_annotations": {"run": 1}}),
                expected=Expected(
                    data={
                        "annotations": {"model": "efish", "processed": True, "run": 1},
                    }
                ),
            ),
            Scenario(
                name="overriding_custom_annotations",
                given=Given(data={"custom_annotations": {"processed": False}}),
                expected=Expected(
                    data={
                        "annotations": {"model": "efish", "processed": False},
                    }
                ),
            ),
        ]

        for scenario in scenarios:
            with self.subTest(scenario.name):
                neuro_data = NeuroData()
                source_proxy = NeoSignalProxy(data_object=self.signal)
                neuro_data.register_proxy(source_proxy)
                filter_criteria = _StubFilterCriteria(inputs={"model": "efish"})
                processor = Processor(neuro_data)

                processor.process(
                    self.strategy,
                    filter_criteria,
                    custom_annotations=scenario.given.data["custom_annotations"],
                )

                all_proxies = list(neuro_data._proxy_registry.values())
                computed = [
                    p for p in all_proxies if isinstance(p, ComputedSignalProxy)
                ]

                self.assertEqual(len(all_proxies), 2)
                self.assertEqual(len(computed), 1)
                self.assertEqual(
                    computed[0].annotations, scenario.expected.data["annotations"]
                )
