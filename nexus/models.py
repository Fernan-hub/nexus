from dataclasses import dataclass
from typing import Any

from nexus.core.interfaces import SignalProxy


@dataclass
class NodeDefinition:
    input_proxies_dict: dict[str, list[SignalProxy]]
    output_annotations: list[dict[str, Any]]
