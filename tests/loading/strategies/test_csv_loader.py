"""Tests for CSVLoader."""

import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import quantities as pq
from neo.core import AnalogSignal

from nexus.loading.strategies.csv_loader import CSVLoader, CSVLoaderConfig
from tests.utils import Expected, Given, Scenario


def _make_config(**kwargs: Any) -> CSVLoaderConfig:
    defaults: dict[str, Any] = {
        "file_path": "test.csv",
        "sampling_rate": 1.0 * pq.kHz,
        "units": pq.mV,
    }
    defaults.update(kwargs)
    return CSVLoaderConfig(**defaults)


class TestCSVLoader(unittest.TestCase):
    """Unit tests for CSVLoader."""

    @pytest.mark.unit
    @pytest.mark.strategy
    @patch("nexus.loading.strategies.csv_loader.AsciiSignalIO")
    def test_load_data(self, mock_io_class: MagicMock) -> None:
        """Test load_data passes config fields to AsciiSignalIO and returns signals."""
        analog = MagicMock(spec=AnalogSignal)
        mock_segment = MagicMock()
        mock_segment.analogsignals = [analog]
        mock_segment.irregularlysampledsignals = []
        mock_io_class.return_value.read_segment.return_value = mock_segment

        scenarios = [
            Scenario(
                name="default config",
                given=Given(
                    data={
                        "file_path": "a.csv",
                        "delimiter": ",",
                        "use_cols": None,
                        "skip_rows": 0,
                        "time_column": None,
                    }
                ),
                expected=Expected(data={"signal_count": 1}),
            ),
            Scenario(
                name="custom delimiter",
                given=Given(
                    data={
                        "file_path": "b.csv",
                        "delimiter": ";",
                        "use_cols": None,
                        "skip_rows": 0,
                        "time_column": None,
                    }
                ),
                expected=Expected(data={"signal_count": 1}),
            ),
            Scenario(
                name="with use_cols",
                given=Given(
                    data={
                        "file_path": "c.csv",
                        "delimiter": ",",
                        "use_cols": [0, 2],
                        "skip_rows": 0,
                        "time_column": None,
                    }
                ),
                expected=Expected(data={"signal_count": 1}),
            ),
            Scenario(
                name="with skip_rows",
                given=Given(
                    data={
                        "file_path": "d.csv",
                        "delimiter": ",",
                        "use_cols": None,
                        "skip_rows": 3,
                        "time_column": None,
                    }
                ),
                expected=Expected(data={"signal_count": 1}),
            ),
            Scenario(
                name="with time_column",
                given=Given(
                    data={
                        "file_path": "e.csv",
                        "delimiter": ",",
                        "use_cols": None,
                        "skip_rows": 0,
                        "time_column": 0,
                    }
                ),
                expected=Expected(data={"signal_count": 1}),
            ),
        ]

        for scenario in scenarios:
            with self.subTest(msg=scenario.name):
                mock_io_class.reset_mock()
                d = scenario.given.data
                config = _make_config(
                    file_path=d["file_path"],
                    delimiter=d["delimiter"],
                    use_cols=d["use_cols"],
                    skip_rows=d["skip_rows"],
                    time_column=d["time_column"],
                )
                loader = CSVLoader(config)

                result = loader.load_data()

                self.assertEqual(len(result), scenario.expected.data["signal_count"])
                self.assertIs(result[0], analog)
                mock_io_class.assert_called_once_with(
                    filename=d["file_path"],
                    delimiter=d["delimiter"],
                    usecols=d["use_cols"],
                    skiprows=d["skip_rows"],
                    timecolumn=d["time_column"],
                    sampling_rate=config.sampling_rate,
                    t_start=config.t_start,
                    units=config.units,
                    time_units=config.time_units,
                )
                mock_io_class.return_value.read_segment.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.strategy
    @patch("nexus.loading.strategies.csv_loader.AsciiSignalIO")
    def test_load_data_combines_analog_and_irregular_signals(
        self, mock_io_class: MagicMock
    ) -> None:
        """Test load_data concatenates analogsignals and irregularlysampledsignals."""
        analog = MagicMock(spec=AnalogSignal)
        irregular = MagicMock()
        mock_segment = MagicMock()
        mock_segment.analogsignals = [analog]
        mock_segment.irregularlysampledsignals = [irregular]
        mock_io_class.return_value.read_segment.return_value = mock_segment

        loader = CSVLoader(_make_config())

        result = loader.load_data()

        self.assertEqual(len(result), 2)
        self.assertIs(result[0], analog)
        self.assertIs(result[1], irregular)


class TestIntegrationCSVLoader(unittest.TestCase):
    """Integration tests for CSVLoader with real file I/O."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @pytest.mark.integration
    @pytest.mark.strategy
    def test_load_data(self) -> None:
        """Test load_data reads correct values across different file configurations."""
        scenarios = [
            Scenario(
                name="comma-delimited single column",
                given=Given(
                    data={
                        "file_content": "1.5\n2.3\n1.8\n",
                        "delimiter": ",",
                        "skip_rows": 0,
                        "use_cols": None,
                        "time_column": None,
                        "units": pq.mV,
                    }
                ),
                expected=Expected(
                    data={
                        "signal_count": 1,
                        "values": [[1.5], [2.3], [1.8]],
                        "units": pq.mV,
                    }
                ),
            ),
            Scenario(
                name="semicolon-delimited two columns",
                given=Given(
                    data={
                        "file_content": "1.0;4.0\n2.0;5.0\n3.0;6.0\n",
                        "delimiter": ";",
                        "skip_rows": 0,
                        "use_cols": None,
                        "time_column": None,
                        "units": pq.mV,
                    }
                ),
                expected=Expected(
                    data={
                        "signal_count": 2,
                        "values": [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],
                        "units": pq.mV,
                    }
                ),
            ),
            Scenario(
                name="skip_rows skips first data row",
                given=Given(
                    data={
                        "file_content": "99.9\n1.5\n2.3\n1.8\n",
                        "delimiter": ",",
                        "skip_rows": 1,
                        "use_cols": None,
                        "time_column": None,
                        "units": pq.mV,
                    }
                ),
                expected=Expected(
                    data={
                        "signal_count": 1,
                        "values": [[1.5], [2.3], [1.8]],
                        "units": pq.mV,
                    }
                ),
            ),
            Scenario(
                name="use_cols selects subset of columns",
                given=Given(
                    data={
                        "file_content": "1.0,2.0,3.0\n1.1,2.1,3.1\n1.2,2.2,3.2\n",
                        "delimiter": ",",
                        "skip_rows": 0,
                        "use_cols": [0, 2],
                        "time_column": None,
                        "units": pq.mV,
                    }
                ),
                expected=Expected(
                    data={
                        "signal_count": 2,
                        "values": [[1.0, 3.0], [1.1, 3.1], [1.2, 3.2]],
                        "units": pq.mV,
                    }
                ),
            ),
            Scenario(
                name="microvolts units",
                given=Given(
                    data={
                        "file_content": "1.5\n2.3\n1.8\n",
                        "delimiter": ",",
                        "skip_rows": 0,
                        "use_cols": None,
                        "time_column": None,
                        "units": pq.uV,
                    }
                ),
                expected=Expected(
                    data={
                        "signal_count": 1,
                        "values": [[1.5], [2.3], [1.8]],
                        "units": pq.uV,
                    }
                ),
            ),
        ]

        for scenario in scenarios:
            with self.subTest(msg=scenario.name):
                d = scenario.given.data
                e = scenario.expected.data
                csv_file = self.temp_path / f"{scenario.name.replace(' ', '_')}.csv"
                csv_file.write_text(d["file_content"])

                config = CSVLoaderConfig(
                    file_path=str(csv_file),
                    sampling_rate=1.0 * pq.kHz,
                    units=d["units"],
                    delimiter=d["delimiter"],
                    skip_rows=d["skip_rows"],
                    use_cols=d["use_cols"],
                    time_column=d["time_column"],
                )
                loader = CSVLoader(config)

                result = loader.load_data()

                self.assertEqual(len(result), e["signal_count"])
                for sig in result:
                    self.assertEqual(sig.units, e["units"])
                values_matrix = np.column_stack(
                    [sig.magnitude.flatten() for sig in result]
                )
                np.testing.assert_array_almost_equal(values_matrix, e["values"])

    @pytest.mark.integration
    @pytest.mark.strategy
    def test_load_data_raises_when_file_not_found(self) -> None:
        """Test that load_data raises when the file does not exist."""
        config = CSVLoaderConfig(
            file_path=str(self.temp_path / "nonexistent.csv"),
            sampling_rate=1.0 * pq.kHz,
            units=pq.mV,
        )
        loader = CSVLoader(config)

        with self.assertRaises((FileNotFoundError, OSError)):
            loader.load_data()
