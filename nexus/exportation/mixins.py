"""Shared plot configuration mixin for figure-based exportation strategies."""

from dataclasses import dataclass


@dataclass
class PlotMixin:
    """Mixin dataclass that adds common matplotlib figure options to a config.

    Intended to be mixed into ExportationStrategyConfig subclasses that render
    figures. All fields are optional; unset fields defer to library defaults or
    the AnalysisResult metadata.

    Parameters
    ----------
    fig_title : str or None, optional
        Title rendered above the figure. Defaults to None (no title).
    fig_size : tuple of (float, float) or None, optional
        Figure dimensions in inches as (width, height). Defaults to None.
    x_label : str or None, optional
        Label for the x-axis. When None, the exporter may use result metadata.
    y_label : str or None, optional
        Label for the y-axis. When None, the exporter may use result metadata.
    tight_layout : bool, optional
        Whether to call plt.tight_layout() before saving. Defaults to True.
    """

    fig_title: str | None = None
    fig_size: tuple[float, float] | None = None
    x_label: str | None = None
    y_label: str | None = None
    tight_layout: bool = True
