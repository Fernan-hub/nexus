"""Tests for NeoSignalProxy and ComputedSignalProxy."""

import unittest
from unittest.mock import MagicMock

import pytest
from neo.io.proxyobjects import BaseProxy

from nexus.core.operation_node import OperationNode
from nexus.core.proxies import ComputedSignalProxy, NeoSignalProxy
from tests.utils import Expected, Given, Scenario


class TestNeoSignalProxy(unittest.TestCase):
    """Unit tests for NeoSignalProxy."""

    @pytest.mark.unit
    @pytest.mark.core
    def test_init(self) -> None:
        """Test init stores annotations and assigns a unique id for both sources."""
        neo_proxy = MagicMock(spec=BaseProxy)
        neo_proxy.annotations = {"model": "efish"}
        data_object = MagicMock()
        data_object.annotations = {"model": "efish"}

        scenarios = [
            Scenario(
                name="from neo proxy",
                given=Given(data={"kwargs": {"neo_proxy": neo_proxy}}),
                expected=Expected(data={"annotations": {"model": "efish"}}),
            ),
            Scenario(
                name="from data object",
                given=Given(data={"kwargs": {"data_object": data_object}}),
                expected=Expected(data={"annotations": {"model": "efish"}}),
            ),
        ]

        for scenario in scenarios:
            with self.subTest(scenario.name):
                proxy = NeoSignalProxy(**scenario.given.data["kwargs"])

                self.assertEqual(
                    proxy.annotations, scenario.expected.data["annotations"]
                )
                self.assertIsInstance(proxy.id, str)

    @pytest.mark.unit
    @pytest.mark.core
    def test_init_raises_value_error_when_neither_provided(self) -> None:
        """Test init raises ValueError when neither argument is given."""
        with self.assertRaises(ValueError):
            NeoSignalProxy()

    @pytest.mark.unit
    @pytest.mark.core
    def test_load(self) -> None:
        """Test load returns the correct signal from both input sources."""
        signal = MagicMock()
        neo_proxy = MagicMock(spec=BaseProxy)
        neo_proxy.annotations = {}
        neo_proxy.load.return_value = signal
        data_object = MagicMock()

        scenarios = [
            Scenario(
                name="from neo proxy",
                given=Given(data={"kwargs": {"neo_proxy": neo_proxy}}),
                expected=Expected(data={"result": signal}),
            ),
            Scenario(
                name="from data object",
                given=Given(data={"kwargs": {"data_object": data_object}}),
                expected=Expected(data={"result": data_object}),
            ),
        ]

        for scenario in scenarios:
            with self.subTest(scenario.name):
                proxy = NeoSignalProxy(**scenario.given.data["kwargs"])

                result = proxy.load()

                self.assertIs(result, scenario.expected.data["result"])

    @pytest.mark.unit
    @pytest.mark.core
    def test_load_caches_result(self) -> None:
        """Test that load caches the result and calls neo_proxy.load only once."""
        neo_proxy = MagicMock(spec=BaseProxy)
        neo_proxy.annotations = {}
        neo_proxy.load.return_value = MagicMock()

        proxy = NeoSignalProxy(neo_proxy=neo_proxy)

        first = proxy.load()
        second = proxy.load()

        self.assertIs(first, second)
        neo_proxy.load.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.core
    def test_load_sets_name_from_annotations(self) -> None:
        """Test that load sets the signal name from the name annotation."""
        signal = MagicMock()
        neo_proxy = MagicMock(spec=BaseProxy)
        neo_proxy.annotations = {"name": "voltage_ch1"}
        neo_proxy.load.return_value = signal

        proxy = NeoSignalProxy(neo_proxy=neo_proxy)

        result = proxy.load()

        self.assertEqual(result.name, "voltage_ch1")

    @pytest.mark.unit
    @pytest.mark.core
    def test_load_does_not_set_name_when_annotation_absent(self) -> None:
        """Test load does not overwrite signal name when name annotation is absent."""
        signal = MagicMock()
        signal.name = "original"
        neo_proxy = MagicMock(spec=BaseProxy)
        neo_proxy.annotations = {}
        neo_proxy.load.return_value = signal

        proxy = NeoSignalProxy(neo_proxy=neo_proxy)

        result = proxy.load()

        self.assertEqual(result.name, "original")

    @pytest.mark.unit
    @pytest.mark.core
    def test_id_is_unique(self) -> None:
        """Test that each proxy instance receives a unique id."""
        signal_a = MagicMock()
        signal_b = MagicMock()

        proxy_a = NeoSignalProxy(data_object=signal_a)
        proxy_b = NeoSignalProxy(data_object=signal_b)

        self.assertNotEqual(proxy_a.id, proxy_b.id)


class TestComputedSignalProxy(unittest.TestCase):
    """Unit tests for ComputedSignalProxy."""

    @pytest.mark.unit
    @pytest.mark.core
    def test_annotations_stored(self) -> None:
        """Test that annotations are stored on the proxy."""
        operation_node = MagicMock(spec=OperationNode)
        annotations = {"model": "efish", "filtered": True}

        proxy = ComputedSignalProxy(
            operation_node=operation_node,
            output_index=0,
            annotations=annotations,
        )

        self.assertEqual(proxy.annotations, annotations)

    @pytest.mark.unit
    @pytest.mark.core
    def test_load_returns_output_at_index(self) -> None:
        """Test that load returns the output at the correct index."""
        signal_a = MagicMock()
        signal_b = MagicMock()
        operation_node = MagicMock(spec=OperationNode)
        operation_node.compute.return_value = [signal_a, signal_b]

        proxy = ComputedSignalProxy(
            operation_node=operation_node,
            output_index=1,
            annotations={},
        )

        result = proxy.load()

        self.assertIs(result, signal_b)
        operation_node.compute.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.core
    def test_load_caches_result(self) -> None:
        """Test that load caches the result and calls compute only once."""
        operation_node = MagicMock(spec=OperationNode)
        operation_node.compute.return_value = [MagicMock()]

        proxy = ComputedSignalProxy(
            operation_node=operation_node,
            output_index=0,
            annotations={},
        )

        first = proxy.load()
        second = proxy.load()

        self.assertIs(first, second)
        operation_node.compute.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.core
    def test_id_is_unique(self) -> None:
        """Test that each proxy instance receives a unique id."""
        node_a = MagicMock(spec=OperationNode)
        node_b = MagicMock(spec=OperationNode)

        proxy_a = ComputedSignalProxy(
            operation_node=node_a, output_index=0, annotations={}
        )
        proxy_b = ComputedSignalProxy(
            operation_node=node_b, output_index=0, annotations={}
        )

        self.assertNotEqual(proxy_a.id, proxy_b.id)
