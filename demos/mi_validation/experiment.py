from pathlib import Path

import quantities as pq

from nexus.core import Analyzer, NeuroData, Processor, ResultsExporter
from nexus.analysis.estimators import DiscreteMutualInformationConfig
from nexus.analysis.strategies import MutualInformationStrategy
from nexus.annotation.strategies import GeneratorAnnotator
from nexus.exportation.strategies import (
    CSVExporter,
    CSVExporterConfig,
    HeatMapExporter,
    HeatMapExporterConfig,
)
from nexus.loading.strategies import CSVLoader, CSVLoaderConfig
from nexus.processing.strategies import StandardizationStrategy

INPUT_FILE_PATH = Path(__file__).parent / "input" / "signals.csv"
OUTPUT_DIR_PATH = Path(__file__).parent / "output"
OUTPUT_DATA_DIR_PATH = OUTPUT_DIR_PATH / "data"
OUTPUT_PLOT_DIR_PATH = OUTPUT_DIR_PATH / "plots"

# Column layout: identical_1, identical_2 (same binary sequence), independent (different seed).
# Expected MI: identical pair H(X) ~= 0.69 nats; any pair with independent ~= 0 nats.
SIGNAL_NAMES = ["identical_1", "identical_2", "independent"]

# Loading phase - one column per proxy so each signal gets its own annotation
neuro_data = NeuroData()

ann_gen = ({"name": name} for name in SIGNAL_NAMES)
annotator = GeneratorAnnotator(iter(ann_gen), {"name": "unknown"})
loader_config = CSVLoaderConfig(
    file_path=str(INPUT_FILE_PATH),
    sampling_rate=1 * pq.kHz,
    units=pq.mV,
    delimiter=",",
)
neuro_data.load_from_file(CSVLoader(loader_config), annotator)

# Processing phase - standardize all signals (zero mean, unit variance)
processor = Processor(neuro_data)
std_strategy = StandardizationStrategy()
processor.process(std_strategy, std_strategy.filter_criteria_type())

# Analysis phase - discrete MI (in nats, base e)
analyzer = Analyzer(neuro_data)
mi_config = DiscreteMutualInformationConfig()
mi_strategy = MutualInformationStrategy(mi_config)
mi_criteria = mi_strategy.filter_criteria_type(
    data_x=[{"standardized": True}],
    data_y=[{"standardized": True}],
)
mi_result = analyzer.analyze_data(mi_strategy, mi_criteria)

# Exportation phase
OUTPUT_DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_PLOT_DIR_PATH.mkdir(parents=True, exist_ok=True)

mi_exporter = ResultsExporter(mi_result)
mi_exporter.export_result(
    CSVExporter(
        CSVExporterConfig(
            output_file_path=str(OUTPUT_DATA_DIR_PATH / "mutual_information.csv")
        )
    )
)
mi_exporter.export_result(
    HeatMapExporter(
        HeatMapExporterConfig(
            output_file_path=str(
                OUTPUT_PLOT_DIR_PATH / "mutual_information_heatmap.png"
            ),
            fig_title="Mutual Information Heatmap (discrete, nats)",
            tight_layout=True,
        )
    )
)
