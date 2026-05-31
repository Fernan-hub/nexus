"""Tests for OperationNode."""

import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest
import quantities as pq
from neo.core import AnalogSignal
from neo.core.dataobject import DataObject

from nexus.core.interfaces import SignalProxy
from nexus.core.operation_node import OperationNode
from nexus.core.proxies import NeoSignalProxy
from nexus.processing.interfaces import ProcessingStrategy


class TestOperationNode(unittest.TestCase):
    """Unit tests for OperationNode."""

    def setUp(self) -> None:
        self.proxy = MagicMock(spec=SignalProxy)
        self.strategy = MagicMock(spec=ProcessingStrategy)
        self.mock_data_input_type = MagicMock()
        self.strategy.data_input_type = self.mock_data_input_type

    @pytest.mark.unit
    @pytest.mark.core
    def test_compute(self) -> None:
        signal_in = MagicMock()
        signal_out = MagicMock()
        self.proxy.load.return_value = signal_in
        self.strategy.apply.return_value = [signal_out]
        node = OperationNode(
            input_proxies={"inputs": [self.proxy]},
            processing_strategy=self.strategy,
        )

        result = node.compute()

        self.assertEqual(result, [signal_out])
        self.proxy.load.assert_called_once()
        self.mock_data_input_type.assert_called_once_with(inputs=[signal_in])
        self.strategy.apply.assert_called_once_with(
            self.mock_data_input_type.return_value
        )

    @pytest.mark.unit
    @pytest.mark.core
    def test_compute_caches_result(self) -> None:
        self.proxy.load.return_value = MagicMock()
        self.strategy.apply.return_value = [MagicMock()]
        node = OperationNode(
            input_proxies={"inputs": [self.proxy]},
            processing_strategy=self.strategy,
        )

        first = node.compute()
        second = node.compute()

        self.assertIs(first, second)
        self.strategy.apply.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.core
    def test_compute_with_multiple_input_groups(self) -> None:
        proxy_a = MagicMock(spec=SignalProxy)
        signal_a = MagicMock()
        proxy_a.load.return_value = signal_a

        proxy_b = MagicMock(spec=SignalProxy)
        signal_b = MagicMock()
        proxy_b.load.return_value = signal_b

        self.strategy.apply.return_value = [MagicMock()]
        node = OperationNode(
            input_proxies={"source": [proxy_a], "target": [proxy_b]},
            processing_strategy=self.strategy,
        )

        node.compute()

        self.mock_data_input_type.assert_called_once_with(
            source=[signal_a], target=[signal_b]
        )


@dataclass
class _PassthroughDataInput:
    inputs: list


class _PassthroughStrategy(ProcessingStrategy):
    """Minimal pass-through strategy for integration testing."""

    supported_data_object_types = [AnalogSignal]

    @property
    def filter_criteria_type(self):  # type: ignore[override]
        return None

    @property
    def proxy_input_type(self):  # type: ignore[override]
        return None

    @property
    def data_input_type(self):  # type: ignore[override]
        return _PassthroughDataInput

    def infer_execution_plan(self, input_proxies):  # type: ignore[override]
        return []

    def apply(self, input_data: _PassthroughDataInput) -> list[DataObject]:
        return list(input_data.inputs)


class TestIntegrationOperationNode(unittest.TestCase):
    """Integration tests for OperationNode with real proxies and strategy."""

    def setUp(self) -> None:
        self.signal = AnalogSignal(
            np.array([[1.0], [2.0]]) * pq.mV,
            sampling_rate=1.0 * pq.kHz,
        )
        self.proxy = NeoSignalProxy(data_object=self.signal)
        self.strategy = _PassthroughStrategy()
        self.node = OperationNode(
            input_proxies={"inputs": [self.proxy]},
            processing_strategy=self.strategy,
        )

    @pytest.mark.integration
    @pytest.mark.core
    def test_compute(self) -> None:
        result = self.node.compute()

        self.assertEqual(len(result), 1)
        self.assertIs(result[0], self.signal)

    @pytest.mark.integration
    @pytest.mark.core
    def test_compute_caches_real_result(self) -> None:
        first = self.node.compute()
        second = self.node.compute()

        self.assertIs(first, second)
