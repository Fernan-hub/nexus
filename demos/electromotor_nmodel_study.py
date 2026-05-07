from pathlib import Path

import quantities as pq

from nexus.core import Analyzer, NeuroData, Processor, ResultsExporter
from nexus.analysis.estimators import DiscreteMutualInformationConfig
from nexus.analysis.strategies import MutualInformationStrategy
from nexus.annotation.strategies import GeneratorAnnotator
from nexus.exportation.strategies import (
    HeatMapExporter,
    HeatMapExporterConfig,
    CSVExporter,
    CSVExporterConfig,
)
from nexus.loading.strategies import CSVLoader, CSVLoaderConfig
from nexus.processing.strategies import (
    BandpassFilterStrategy,
    BinnedSpikeTrainStrategy,
    SpikeExtractionStrategy,
)

INPUT_DIR_NAME = "input"
INPUT_DIR_PATH = Path(__file__).parent.joinpath(INPUT_DIR_NAME)

OUTPUT_DIR_NAME = "output"
OUTPUT_DIR_PATH = Path(__file__).parent.joinpath(OUTPUT_DIR_NAME)

OUTPUT_PLOT_DIR_NAME = "plots"
OUTPUT_PLOT_DIR_PATH = OUTPUT_DIR_PATH.joinpath(OUTPUT_PLOT_DIR_NAME)

OUTPUT_DATA_DIR_NAME = "data"
OUTPUT_DATA_DIR_PATH = OUTPUT_DIR_PATH.joinpath(OUTPUT_DATA_DIR_NAME)

PATTERNS = ["acc", "cess", "rasp", "sca"]
SGA_INPUT_FILE_NAMES = [f"SGA_{pattern}" for pattern in PATTERNS]

SGA_ACC_FILE_NAME = "SGA_acc"
SGA_CESS_FILE_NAME = "SGA_cess"
SGA_RASP_FILE_NAME = "SGA_rasp"
SGA_SCA_FILE_NAME = "SGA_sca"

CURRENT_NAMES = ["E_DP", "E_PCN", "I_DP", "I_PCN", "E_CDP"]
VOLTAGE_NAMES = ["VPd", "DP", "PCN", "CN"]


# Loading phase

neuro_data = NeuroData()

for pattern, file_name in zip(PATTERNS, SGA_INPUT_FILE_NAMES):
    base_ann = {"model": "SGA", "pattern": pattern}
    default_ann = {"name": "unknown", "type": "unknown", **base_ann}

    # Load current data
    ann_gen = ({"name": name, "type": "current", **base_ann} for name in CURRENT_NAMES)
    annotator = GeneratorAnnotator(ann_gen, default_ann)

    loader_config = CSVLoaderConfig(
        file_path=str(INPUT_DIR_PATH.joinpath(file_name)),
        sampling_rate=100 * pq.kHz,
        units=pq.mA,
        delimiter=" ",
        use_cols=list(range(len(CURRENT_NAMES))),
    )
    loader = CSVLoader(loader_config)

    neuro_data.load_from_file(loader, annotator)

    # Load voltage data
    ann_gen = ({"name": name, "type": "voltage", **base_ann} for name in VOLTAGE_NAMES)
    annotator = GeneratorAnnotator(ann_gen, default_ann)

    loader_config = CSVLoaderConfig(
        file_path=str(INPUT_DIR_PATH.joinpath(file_name)),
        sampling_rate=100 * pq.kHz,
        units=pq.mV,
        delimiter=" ",
        use_cols=list(
            range(len(CURRENT_NAMES), len(CURRENT_NAMES) + len(VOLTAGE_NAMES))
        ),
    )
    loader = CSVLoader(loader_config)

    neuro_data.load_from_file(loader, annotator)

# Processing phase

processor = Processor(neuro_data)

bandpass_filter_strategy = BandpassFilterStrategy()
bandpass_filter_filter_criteria = bandpass_filter_strategy.filter_criteria_type()
processor.process(bandpass_filter_strategy, bandpass_filter_filter_criteria)

spike_extraction_strategy = SpikeExtractionStrategy(threshold=0.5 * pq.mV)
spike_extraction_filter_criteria = spike_extraction_strategy.filter_criteria_type(
    [{"type": "voltage"}, {"filtered": True}]
)
processor.process(spike_extraction_strategy, spike_extraction_filter_criteria)

binned_spike_train_strategy = BinnedSpikeTrainStrategy(bin_size=1 * pq.ms)
binned_spike_train_filter_criteria = binned_spike_train_strategy.filter_criteria_type(
    [{"type": "voltage"}, {"spike": True}]
)
processor.process(binned_spike_train_strategy, binned_spike_train_filter_criteria)

# Analysis phase

analyzer = Analyzer(neuro_data)

mutual_info_config = DiscreteMutualInformationConfig()
mutual_info_strategy = MutualInformationStrategy(mutual_info_config)
mutual_info_filter_criteria = mutual_info_strategy.filter_criteria_type(
    data_x=[
        {"model": "SGA"},
        {"pattern": "acc"},
        {"type": "voltage"},
        {"spike": True},
        {"binned": True},
    ],
    data_y=[
        {"model": "SGA"},
        {"pattern": "acc"},
        {"type": "voltage"},
        {"spike": True},
        {"binned": True},
    ],
)
mutual_info_result = analyzer.analyze_data(
    mutual_info_strategy, mutual_info_filter_criteria
)

# Exportation phase

OUTPUT_PLOT_DIR_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)

exporter = ResultsExporter(mutual_info_result)

heat_map_exporter_config = HeatMapExporterConfig(
    output_file_path=str(
        OUTPUT_PLOT_DIR_PATH.joinpath("mutual_information_heatmap.png")
    ),
    fig_title="Mutual Information Heatmap",
    tight_layout=True,
)
heat_map_exporter = HeatMapExporter(heat_map_exporter_config)
exporter.export_result(heat_map_exporter)

csv_exporter_config = CSVExporterConfig(
    output_file_path=str(OUTPUT_DATA_DIR_PATH.joinpath("mutual_information.csv"))
)
csv_exporter = CSVExporter(csv_exporter_config)
exporter.export_result(csv_exporter)
