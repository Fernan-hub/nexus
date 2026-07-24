"""Tests for NWBLoader."""

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import quantities as pq
from neo.core import AnalogSignal, Block, Segment
from neo.io import NWBIO as _NWBIO
from neo.io.proxyobjects import BaseProxy

from nexus.loading.strategies.nwb_loader import NWBLoader, NWBLoaderConfig
from tests.utils import Expected, Given, Scenario


def _make_nwb_file(file_path: Path, signals: list[AnalogSignal]) -> None:
    block = Block(name="block0")
    segment = Segment(name="segment0")
    block.segments.append(segment)
    segment.block = block
    for signal in signals:
        segment.analogsignals.append(signal)
        signal.segment = segment
    writer = _NWBIO(
        str(file_path), mode="w", session_start_time=datetime.now(timezone.utc)
    )
    writer.write_all_blocks([block], validate=False)


class TestNWBLoader(unittest.TestCase):
    """Unit tests for NWBLoader."""

    @pytest.mark.unit
    @pytest.mark.strategy
    @patch("nexus.loading.strategies.nwb_loader.NWBIO")
    def test_load_data(self, mock_io_class: MagicMock) -> None:
        """Test load_data opens the file with NWBIO and reads all blocks lazily."""
        proxy = MagicMock(spec=BaseProxy)
        mock_segment = MagicMock()
        mock_segment.analogsignals = [proxy]
        mock_segment.irregularlysampledsignals = []
        mock_segment.name = "seg"
        mock_segment.annotations = {}
        mock_block = MagicMock()
        mock_block.segments = [mock_segment]
        mock_block.name = "block"
        mock_block.annotations = {}
        mock_io_class.return_value.read_all_blocks.return_value = [mock_block]

        loader = NWBLoader(NWBLoaderConfig(file_path="recording.nwb"))

        result = loader.load_data()

        mock_io_class.assert_called_once_with(filename="recording.nwb")
        mock_io_class.return_value.read_all_blocks.assert_called_once_with(lazy=False)
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], proxy)


class TestIntegrationNWBLoader(unittest.TestCase):
    """Integration tests for NWBLoader."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @pytest.mark.integration
    @pytest.mark.strategy
    def test_load_data(self) -> None:
        """Test load_data returns one AnalogSignal per channel with correct values."""
        scenarios = [
            Scenario(
                name="single signal",
                given=Given(
                    data={
                        "signals": [
                            AnalogSignal(
                                np.array([[1.0], [2.0], [3.0]]) * pq.mV,
                                sampling_rate=1000 * pq.Hz,
                                name="ch0",
                                t_start=0 * pq.s,
                            ),
                        ],
                    }
                ),
                expected=Expected(
                    data={
                        "signal_count": 1,
                        "units": pq.mV,
                        "values": [np.array([[1.0], [2.0], [3.0]])],
                    }
                ),
            ),
            Scenario(
                name="multiple signals",
                given=Given(
                    data={
                        "signals": [
                            AnalogSignal(
                                np.array([[1.0], [2.0]]) * pq.uV,
                                sampling_rate=1000 * pq.Hz,
                                name="ch0",
                                t_start=0 * pq.s,
                            ),
                            AnalogSignal(
                                np.array([[3.0], [4.0]]) * pq.uV,
                                sampling_rate=1000 * pq.Hz,
                                name="ch1",
                                t_start=0 * pq.s,
                            ),
                        ],
                    }
                ),
                expected=Expected(
                    data={
                        "signal_count": 2,
                        "units": pq.uV,
                        "values": [np.array([[1.0], [2.0]]), np.array([[3.0], [4.0]])],
                    }
                ),
            ),
        ]

        for scenario in scenarios:
            with self.subTest(msg=scenario.name):
                nwb_path = self.temp_path / f"{scenario.name.replace(' ', '_')}.nwb"
                _make_nwb_file(nwb_path, scenario.given.data["signals"])

                loader = NWBLoader(NWBLoaderConfig(file_path=str(nwb_path)))
                result = loader.load_data()

                self.assertEqual(len(result), scenario.expected.data["signal_count"])
                for signal, expected_values in zip(
                    result, scenario.expected.data["values"]
                ):
                    self.assertIsInstance(signal, AnalogSignal)
                    self.assertEqual(signal.units, scenario.expected.data["units"])
                    np.testing.assert_array_almost_equal(
                        signal.magnitude, expected_values
                    )

    @pytest.mark.integration
    @pytest.mark.strategy
    def test_load_data_raises_when_file_not_found(self) -> None:
        """Test that load_data raises when the NWB file does not exist."""
        loader = NWBLoader(NWBLoaderConfig(file_path="/nonexistent/path/recording.nwb"))

        with self.assertRaises((FileNotFoundError, OSError)):
            loader.load_data()
