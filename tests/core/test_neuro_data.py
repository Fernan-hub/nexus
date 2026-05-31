"""Tests for NeuroData."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import quantities as pq
from neo.core import AnalogSignal, SpikeTrain
from neo.io.proxyobjects import BaseProxy

from nexus.annotation.interfaces import AnnotationStrategy
from nexus.core.interfaces import SignalProxy
from nexus.core.neuro_data import NeuroData
from nexus.core.proxies import NeoSignalProxy
from nexus.loading.interfaces import DataLoader
from tests.utils import Expected, Given, Scenario


class TestNeuroData(unittest.TestCase):
    """Unit tests for NeuroData."""

    def setUp(self) -> None:
        self.neuro_data = NeuroData()

    def _make_to_neo_block_fixtures(
        self, MockSegment: MagicMock, MockBlock: MagicMock
    ) -> tuple[MagicMock, MagicMock, MagicMock, MagicMock, MagicMock, MagicMock]:
        signal = MagicMock(spec=AnalogSignal)
        spike = MagicMock(spec=SpikeTrain)
        proxy_sig = MagicMock(spec=SignalProxy)
        proxy_sig.load.return_value = signal
        proxy_spk = MagicMock(spec=SignalProxy)
        proxy_spk.load.return_value = spike
        mock_block = MagicMock()
        mock_block.segments = []
        MockBlock.return_value = mock_block
        mock_seg = MagicMock()
        mock_seg.analogsignals = []
        mock_seg.spiketrains = []
        MockSegment.return_value = mock_seg
        return signal, spike, proxy_sig, proxy_spk, mock_block, mock_seg

    @pytest.mark.unit
    @pytest.mark.core
    def test_register_proxy(self) -> None:
        """Proxy is stored in the registry under its id."""
        proxy = MagicMock(spec=SignalProxy)
        proxy.id = "abc123"

        self.neuro_data.register_proxy(proxy)

        self.assertIn("abc123", self.neuro_data._proxy_registry)
        self.assertIs(self.neuro_data._proxy_registry["abc123"], proxy)

    @pytest.mark.unit
    @pytest.mark.core
    def test_get_proxy_by_id(self) -> None:
        """Returns the registered proxy by id, or None when not found."""
        proxy = MagicMock(spec=SignalProxy)
        self.neuro_data._proxy_registry["abc123"] = proxy

        scenarios = [
            Scenario(
                name="found",
                given=Given(data={"proxy_id": "abc123"}),
                expected=Expected(data={"result": proxy}),
            ),
            Scenario(
                name="not_found",
                given=Given(data={"proxy_id": "nonexistent"}),
                expected=Expected(data={"result": None}),
            ),
        ]

        for scenario in scenarios:
            with self.subTest(scenario.name):
                result = self.neuro_data.get_proxy_by_id(
                    scenario.given.data["proxy_id"]
                )

                self.assertIs(result, scenario.expected.data["result"])

    @pytest.mark.unit
    @pytest.mark.core
    @patch("nexus.core.neuro_data.filterdata")
    def test_get_proxies_by_criteria(self, mock_filterdata: MagicMock) -> None:
        """Delegates filtering to filterdata regardless of whether criteria is None."""
        proxy = MagicMock(spec=SignalProxy)
        self.neuro_data._proxy_registry["abc123"] = proxy
        mock_filterdata.return_value = [proxy]

        scenarios = [
            Scenario(
                name="with_criteria",
                given=Given(data={"criteria": {"model": "efish"}}),
                expected=Expected(data={"result": [proxy]}),
            ),
            Scenario(
                name="with_none_criteria",
                given=Given(data={"criteria": None}),
                expected=Expected(data={"result": [proxy]}),
            ),
        ]

        for scenario in scenarios:
            with self.subTest(scenario.name):
                result = self.neuro_data.get_proxies_by_criteria(
                    scenario.given.data["criteria"]
                )

                self.assertEqual(result, scenario.expected.data["result"])

        self.assertEqual(mock_filterdata.call_count, 2)

    @pytest.mark.unit
    @pytest.mark.core
    @patch("nexus.core.neuro_data.filterdata")
    def test_get_proxies_by_criteria_dict(self, mock_filterdata: MagicMock) -> None:
        """Returns a dict mapping each key to the proxies matching its criteria."""
        proxy_a = MagicMock(spec=SignalProxy)
        proxy_b = MagicMock(spec=SignalProxy)
        mock_filterdata.side_effect = [[proxy_a], [proxy_b]]

        result = self.neuro_data.get_proxies_by_criteria_dict(
            {"source": {"type": "A"}, "target": {"type": "B"}}
        )

        self.assertEqual(result["source"], [proxy_a])
        self.assertEqual(result["target"], [proxy_b])
        self.assertEqual(mock_filterdata.call_count, 2)

    @pytest.mark.unit
    @pytest.mark.core
    def test_load_from_file_with_base_proxy(self) -> None:
        """BaseProxy items are wrapped as NeoSignalProxy and annotated."""
        neo_proxy = MagicMock(spec=BaseProxy)
        neo_proxy.annotations = {"model": "efish"}
        data_loader = MagicMock(spec=DataLoader)
        data_loader.load_data.return_value = [neo_proxy]
        annotation_strategy = MagicMock(spec=AnnotationStrategy)

        self.neuro_data.load_from_file(data_loader, annotation_strategy)

        proxies = list(self.neuro_data._proxy_registry.values())
        self.assertEqual(len(proxies), 1)
        self.assertIsInstance(proxies[0], NeoSignalProxy)
        annotation_strategy.annotate.assert_called_once_with(proxies)

    @pytest.mark.unit
    @pytest.mark.core
    def test_load_from_file_with_data_object(self) -> None:
        """DataObject items are wrapped as NeoSignalProxy and annotated."""
        signal = MagicMock(spec=AnalogSignal)
        signal.annotations = {}
        data_loader = MagicMock(spec=DataLoader)
        data_loader.load_data.return_value = [signal]
        annotation_strategy = MagicMock(spec=AnnotationStrategy)

        self.neuro_data.load_from_file(data_loader, annotation_strategy)

        proxies = list(self.neuro_data._proxy_registry.values())
        self.assertEqual(len(proxies), 1)
        self.assertIsInstance(proxies[0], NeoSignalProxy)
        annotation_strategy.annotate.assert_called_once_with(proxies)

    @pytest.mark.unit
    @pytest.mark.core
    def test_load_from_file_skips_unsupported_data(self) -> None:
        """Items that are neither BaseProxy nor DataObject are silently skipped."""
        data_loader = MagicMock(spec=DataLoader)
        data_loader.load_data.return_value = [MagicMock()]
        annotation_strategy = MagicMock(spec=AnnotationStrategy)

        self.neuro_data.load_from_file(data_loader, annotation_strategy)

        self.assertEqual(self.neuro_data._proxy_registry, {})
        annotation_strategy.annotate.assert_called_once_with([])

    @pytest.mark.unit
    @pytest.mark.core
    @patch("nexus.core.neuro_data.Block")
    @patch("nexus.core.neuro_data.Segment")
    def test_to_neo_block_without_criteria(
        self, MockSegment: MagicMock, MockBlock: MagicMock
    ) -> None:
        """All registered proxies are loaded into a single unnamed segment."""
        signal, spike, proxy_sig, proxy_spk, mock_block, mock_seg = (
            self._make_to_neo_block_fixtures(MockSegment, MockBlock)
        )
        self.neuro_data._proxy_registry["sig"] = proxy_sig
        self.neuro_data._proxy_registry["spk"] = proxy_spk

        block = self.neuro_data.to_neo_block()

        self.assertIs(block, mock_block)
        MockSegment.assert_called_with(name=None)
        self.assertCountEqual(mock_block.segments, [mock_seg])
        self.assertCountEqual(mock_seg.analogsignals, [signal])
        self.assertCountEqual(mock_seg.spiketrains, [spike])

    @pytest.mark.unit
    @pytest.mark.core
    @patch("nexus.core.neuro_data.Block")
    @patch("nexus.core.neuro_data.Segment")
    @patch("nexus.core.neuro_data.filterdata")
    def test_to_neo_block_with_criteria_dict(
        self, mock_filterdata: MagicMock, MockSegment: MagicMock, MockBlock: MagicMock
    ) -> None:
        """Proxies matching each criteria key are loaded into a named segment."""
        signal, spike, proxy_sig, proxy_spk, mock_block, mock_seg = (
            self._make_to_neo_block_fixtures(MockSegment, MockBlock)
        )
        mock_filterdata.return_value = [proxy_sig, proxy_spk]

        block = self.neuro_data.to_neo_block(criteria_dict={"source": {"model": "a"}})

        self.assertIs(block, mock_block)
        MockSegment.assert_called_with(name="source")
        self.assertCountEqual(mock_block.segments, [mock_seg])
        self.assertCountEqual(mock_seg.analogsignals, [signal])
        self.assertCountEqual(mock_seg.spiketrains, [spike])

    @pytest.mark.unit
    @pytest.mark.core
    @patch("nexus.core.neuro_data.Block")
    @patch("nexus.core.neuro_data.Segment")
    def test_to_neo_block_skips_unsupported_data(
        self, MockSegment: MagicMock, MockBlock: MagicMock
    ) -> None:
        """Loaded data that is neither AnalogSignal nor SpikeTrain is skipped."""
        proxy = MagicMock(spec=SignalProxy)
        proxy.load.return_value = MagicMock()
        mock_block = MagicMock()
        mock_block.segments = []
        MockBlock.return_value = mock_block
        mock_seg = MagicMock()
        mock_seg.analogsignals = []
        mock_seg.spiketrains = []
        MockSegment.return_value = mock_seg
        self.neuro_data._proxy_registry["xyz"] = proxy

        block = self.neuro_data.to_neo_block()

        self.assertIs(block, mock_block)
        self.assertCountEqual(mock_seg.analogsignals, [])
        self.assertCountEqual(mock_seg.spiketrains, [])


class TestIntegrationNeuroData(unittest.TestCase):
    """Integration tests for NeuroData with real proxies and filtering."""

    def setUp(self) -> None:
        self.neuro_data = NeuroData()

    def _assert_block_segments(
        self, block: object, expected_segments: list[dict]
    ) -> None:
        self.assertEqual(len(block.segments), len(expected_segments))
        segments_by_name = {seg.name: seg for seg in block.segments}
        for expected_seg in expected_segments:
            seg = segments_by_name[expected_seg["name"]]
            self.assertEqual(len(seg.analogsignals), len(expected_seg["analogs"]))
            self.assertEqual(len(seg.spiketrains), len(expected_seg["spikes"]))
            for i, expected_signal in enumerate(expected_seg["analogs"]):
                self.assertIs(seg.analogsignals[i], expected_signal)
            for i, expected_spike in enumerate(expected_seg["spikes"]):
                self.assertIs(seg.spiketrains[i], expected_spike)

    @pytest.mark.integration
    @pytest.mark.core
    def test_register_and_retrieve_proxy(self) -> None:
        """A registered proxy can be retrieved by its id."""
        signal = AnalogSignal(np.array([[1.0]]) * pq.mV, sampling_rate=1.0 * pq.kHz)
        proxy = NeoSignalProxy(data_object=signal)

        self.neuro_data.register_proxy(proxy)
        registered_proxy = self.neuro_data.get_proxy_by_id(proxy.id)

        self.assertIs(registered_proxy, proxy)

    @pytest.mark.integration
    @pytest.mark.core
    def test_get_proxies_by_criteria(self) -> None:
        """Filters proxies by annotation value; returns all when criteria is None."""
        signal_a = AnalogSignal(
            np.array([[1.0]]) * pq.mV,
            sampling_rate=1.0 * pq.kHz,
            model="efish",
        )
        signal_b = AnalogSignal(
            np.array([[2.0]]) * pq.mV,
            sampling_rate=1.0 * pq.kHz,
            model="other",
        )
        proxy_a = NeoSignalProxy(data_object=signal_a)
        proxy_b = NeoSignalProxy(data_object=signal_b)
        self.neuro_data.register_proxy(proxy_a)
        self.neuro_data.register_proxy(proxy_b)

        scenarios = [
            Scenario(
                name="filters_by_annotation",
                given=Given(data={"criteria": {"model": "efish"}}),
                expected=Expected(data={"result": [proxy_a]}),
            ),
            Scenario(
                name="returns_all_when_none",
                given=Given(data={"criteria": None}),
                expected=Expected(data={"result": [proxy_a, proxy_b]}),
            ),
        ]

        for scenario in scenarios:
            with self.subTest(scenario.name):
                result = self.neuro_data.get_proxies_by_criteria(
                    scenario.given.data["criteria"]
                )

                self.assertCountEqual(result, scenario.expected.data["result"])

    @pytest.mark.integration
    @pytest.mark.core
    def test_to_neo_block(self) -> None:
        """Builds a Block with correct segments, analog signals, and spike trains."""
        signal_src = AnalogSignal(
            np.array([[1.0]]) * pq.mV, sampling_rate=1.0 * pq.kHz, role="source"
        )
        signal_tgt = AnalogSignal(
            np.array([[2.0]]) * pq.mV, sampling_rate=1.0 * pq.kHz, role="target"
        )
        spike_train = SpikeTrain([0.1, 0.2] * pq.s, t_stop=1.0 * pq.s, role="target")

        neuro_data = NeuroData()
        neuro_data.register_proxy(NeoSignalProxy(data_object=signal_src))
        neuro_data.register_proxy(NeoSignalProxy(data_object=signal_tgt))
        neuro_data.register_proxy(NeoSignalProxy(data_object=spike_train))

        scenarios = [
            Scenario(
                name="without_criteria",
                given=Given(data={"criteria_dict": None}),
                expected=Expected(
                    data={
                        "segments": [
                            {
                                "name": None,
                                "analogs": [signal_src, signal_tgt],
                                "spikes": [spike_train],
                            }
                        ]
                    }
                ),
            ),
            Scenario(
                name="with_criteria_dict",
                given=Given(
                    data={
                        "criteria_dict": {
                            "source": {"role": "source"},
                            "target": {"role": "target"},
                        }
                    }
                ),
                expected=Expected(
                    data={
                        "segments": [
                            {"name": "source", "analogs": [signal_src], "spikes": []},
                            {
                                "name": "target",
                                "analogs": [signal_tgt],
                                "spikes": [spike_train],
                            },
                        ]
                    }
                ),
            ),
        ]

        for scenario in scenarios:
            with self.subTest(scenario.name):
                block = neuro_data.to_neo_block(
                    criteria_dict=scenario.given.data["criteria_dict"]
                )

                self._assert_block_segments(block, scenario.expected.data["segments"])
