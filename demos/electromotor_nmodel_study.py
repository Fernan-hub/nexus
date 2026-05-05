from pathlib import Path
from itertools import repeat, chain

import quantities as pq

from nexus.core import Analyzer, NeuroData, Processor
from nexus.analysis.estimators import KernelMutualInformationConfig
from nexus.analysis.strategies import MutualInformationStrategy
from nexus.annotation.strategies import GeneratorAnnotator
from nexus.loading.strategies import CSVLoader, CSVLoaderConfig
from nexus.processing.strategies import BandpassFilterStrategy

INPUT_DIR_NAME = "input"
INPUT_DIR_PATH = Path(__file__).parent.joinpath(INPUT_DIR_NAME)

PATTERNS = ["acc", "cess", "rasp", "sca"]
SGA_INPUT_FILE_NAMES = [f"SGA_{pattern}" for pattern in PATTERNS]

SGA_ACC_FILE_NAME = "SGA_acc"
SGA_CESS_FILE_NAME = "SGA_cess"
SGA_RASP_FILE_NAME = "SGA_rasp"
SGA_SCA_FILE_NAME = "SGA_sca"

CURRENT_NAMES = ["E_DP", "E_PCN", "I_DP", "I_PCN", "E_CDP"]
VOLTAGE_NAMES = ["VPd", "DP", "PCN", "CN"]

VAR_DATA_ITER = chain(
    zip(VOLTAGE_NAMES, repeat("voltage")), zip(CURRENT_NAMES, repeat("current"))
)
VAR_DATA = list(VAR_DATA_ITER)

# Loading phase

neuro_data = NeuroData()

for pattern, file_name in zip(PATTERNS, SGA_INPUT_FILE_NAMES):
    base_ann = {"model": "SGA", "pattern": pattern}
    ann_gen = (
        {"name": name, "type": data_type, **base_ann} for name, data_type in VAR_DATA
    )
    default_ann = {"name": "unknown", "type": "unknown", **base_ann}
    annotator = GeneratorAnnotator(ann_gen, default_ann)

    loader_config = CSVLoaderConfig(
        file_path=str(INPUT_DIR_PATH.joinpath(file_name)),
        sampling_rate=100 * pq.kHz,
        units=pq.mV,
        delimiter=" ",
    )
    loader = CSVLoader(loader_config)

    neuro_data.load_from_file(loader, annotator)

# Processing phase

processor = Processor(neuro_data)

bandpass_filter_strategy = BandpassFilterStrategy()
bandpass_filter_filter_criteria = bandpass_filter_strategy.filter_criteria_type()
processor.process(bandpass_filter_strategy, bandpass_filter_filter_criteria)


# Analysis phase

analyzer = Analyzer(neuro_data)

mutual_info_config = KernelMutualInformationConfig()
mutual_info_strategy = MutualInformationStrategy(mutual_info_config)
mutual_info_filter_criteria = mutual_info_strategy.filter_criteria_type(
    data_x=[
        {"model": "SGA"},
        {"pattern": "acc"},
        {"type": "voltage"},
        {"filtered": True},
    ],
    data_y=[
        {"model": "SGA"},
        {"pattern": "acc"},
        {"type": "voltage"},
        {"filtered": True},
    ],
)
mutual_info_result = analyzer.analyze_data(
    mutual_info_strategy, mutual_info_filter_criteria
)
