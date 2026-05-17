from pathlib import Path

import quantities as pq

from nexus.core import Analyzer, NeuroData, Processor, ResultsExporter
from nexus.analysis.estimators import (
    DiscreteMutualInformationConfig,
    DiscreteTransferEntropyConfig,
)
from nexus.analysis.strategies import MutualInformationStrategy, TransferEntropyStrategy
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
    ConcatenationStrategy,
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

CURRENT_NAMES = ["E_DP", "E_PCN", "I_DP", "I_PCN", "E_CDP"]
VOLTAGE_NAMES = ["VPd", "DP", "PCN", "CN"]

BIN_SIZE = 1 * pq.ms

# t_start offsets make the binned spike trains from each pattern contiguous
# so they can be concatenated into a single long sequence.
# Offsets = cumulative bin counts (n_lines // 100) x 1 ms per bin.
# File line counts: acc=200003, cess=100003, rasp=133003, sca=120003.
PATTERN_T_STARTS = {
    "acc": 0 * pq.ms,  # bins: 0
    "cess": 2000 * pq.ms,  # bins: 2000
    "rasp": 3000 * pq.ms,  # bins: 2000 + 1000
    "sca": 4330 * pq.ms,  # bins: 2000 + 1000 + 1330
}


# Loading phase

neuro_data = NeuroData()

for pattern, file_name in zip(PATTERNS, SGA_INPUT_FILE_NAMES):
    t_start = PATTERN_T_STARTS[pattern]
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
        t_start=t_start,
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
        t_start=t_start,
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

binned_spike_train_strategy = BinnedSpikeTrainStrategy(bin_size=BIN_SIZE)
binned_spike_train_filter_criteria = binned_spike_train_strategy.filter_criteria_type(
    [{"type": "voltage"}, {"spike": True}]
)
processor.process(binned_spike_train_strategy, binned_spike_train_filter_criteria)

# Concatenate each voltage channel across all 4 patterns into a single long sequence
concatenation_strategy = ConcatenationStrategy()
for voltage_name in VOLTAGE_NAMES:
    concat_filter_criteria = concatenation_strategy.filter_criteria_type(
        inputs=[{"name": voltage_name}, {"binned": True}, {"type": "voltage"}]
    )
    processor.process(concatenation_strategy, concat_filter_criteria)

# Analysis phase

analyzer = Analyzer(neuro_data)

CONCAT_VOLTAGE_CRITERIA = [
    {"type": "voltage"},
    {"binned": True},
    {"concatenated": True},
]

mutual_info_config = DiscreteMutualInformationConfig()
mutual_info_strategy = MutualInformationStrategy(mutual_info_config)
mutual_info_filter_criteria = mutual_info_strategy.filter_criteria_type(
    data_x=CONCAT_VOLTAGE_CRITERIA,
    data_y=CONCAT_VOLTAGE_CRITERIA,
)
mutual_info_result = analyzer.analyze_data(
    mutual_info_strategy, mutual_info_filter_criteria
)

# prop_time=1 captures a 1 ms propagation delay (DP/PCN -> CN synaptic lag)
transfer_entropy_config = DiscreteTransferEntropyConfig(prop_time=1)
transfer_entropy_strategy = TransferEntropyStrategy(transfer_entropy_config)
transfer_entropy_filter_criteria = transfer_entropy_strategy.filter_criteria_type(
    sources=CONCAT_VOLTAGE_CRITERIA,
    dests=CONCAT_VOLTAGE_CRITERIA,
)
transfer_entropy_result = analyzer.analyze_data(
    transfer_entropy_strategy, transfer_entropy_filter_criteria
)

# Exportation phase

OUTPUT_PLOT_DIR_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)

mi_exporter = ResultsExporter(mutual_info_result)

heat_map_exporter_config = HeatMapExporterConfig(
    output_file_path=str(
        OUTPUT_PLOT_DIR_PATH.joinpath("mutual_information_heatmap.png")
    ),
    fig_title="Mutual Information Heatmap",
    tight_layout=True,
)
heat_map_exporter = HeatMapExporter(heat_map_exporter_config)
mi_exporter.export_result(heat_map_exporter)

csv_exporter_config = CSVExporterConfig(
    output_file_path=str(OUTPUT_DATA_DIR_PATH.joinpath("mutual_information.csv"))
)
csv_exporter = CSVExporter(csv_exporter_config)
mi_exporter.export_result(csv_exporter)

te_exporter = ResultsExporter(transfer_entropy_result)

heat_map_exporter_config = HeatMapExporterConfig(
    output_file_path=str(OUTPUT_PLOT_DIR_PATH.joinpath("transfer_entropy_heatmap.png")),
    fig_title="Transfer Entropy Heatmap (prop_time=1ms)",
    tight_layout=True,
)
heat_map_exporter = HeatMapExporter(heat_map_exporter_config)
te_exporter.export_result(heat_map_exporter)

csv_exporter_config = CSVExporterConfig(
    output_file_path=str(OUTPUT_DATA_DIR_PATH.joinpath("transfer_entropy.csv"))
)
csv_exporter = CSVExporter(csv_exporter_config)
te_exporter.export_result(csv_exporter)
