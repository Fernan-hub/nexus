from dataclasses import dataclass


@dataclass
class PlotMixin:
    fig_title: str | None = None
    fig_size: tuple[float, float] | None = None
    x_label: str | None = None
    y_label: str | None = None
    tight_layout: bool = True
