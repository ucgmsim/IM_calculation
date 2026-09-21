"""Standalone HTML difference reports for intensity measure comparisons.

The benchmark tests compare a stored table of intensity measures against a
freshly calculated one. When they disagree, a bare table of numbers rarely
explains *what* changed, so this module renders the comparison as a single
self-contained HTML file (no network access, no JavaScript charting library):

- headline statistics and a per-intensity-measure-family breakdown,
- inline SVG plots of the differences (spectral ratio plots, benchmark against
  current spectra, a difference histogram and an error percentile curve),
- colour coded tables of every compared value, one table per family, filterable
  down to just the values that fall outside tolerance.

Examples
--------
>>> from IM import diff_report
>>> diff_report.write_diff_report(
...     benchmark, result, Path("diff.html"), title="2024p950420_MWFS_HN_20"
... )  # doctest: +SKIP
"""

import datetime
import html
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

DEFAULT_ATOL = 5e-4
"""Absolute tolerance matching the benchmark tests."""

DEFAULT_RTOL = 0.01
"""Relative tolerance matching the benchmark tests."""

MAX_PLOTTED_SERIES = 8
"""Maximum number of rows drawn in a single plot (the categorical palette size)."""

_clip_counter = 0


def _next_clip_id() -> str:
    """Return a document-unique id for a clip path.

    Returns
    -------
    str
        The identifier.
    """
    global _clip_counter
    _clip_counter += 1
    return f"clip{_clip_counter}"


NOISE_ULP = 4.0
"""Distance in doubles under which a difference is treated as floating point noise."""

_MINOR_LEVEL = 0.05
"""Relative difference above which a cell is shaded a mid tone."""

_MAJOR_LEVEL = 0.20
"""Relative difference above which a cell is shaded the strongest tone."""


@dataclass(frozen=True)
class FamilyAxes:
    """Axis descriptions for a parameterised intensity measure family."""

    parameter_label: str
    """Name of the parameter encoded in the column name (e.g. ``Period``)."""

    parameter_unit: str
    """Unit of that parameter (e.g. ``s``)."""

    value_label: str
    """Label (including unit) for the intensity measure itself."""


FAMILY_AXES: dict[str, FamilyAxes] = {
    "pSA": FamilyAxes("Period", "s", "pSA (g)"),
    "FAS": FamilyAxes("Frequency", "Hz", "FAS (cm/s)"),
    "SNR": FamilyAxes("Frequency", "Hz", "SNR"),
}
"""Axis metadata for the families that carry a parameter in their column name."""

DEFAULT_FAMILY_AXES = FamilyAxes("Parameter", "", "Value")
"""Axis metadata used for parameterised families we do not know about."""

_COLUMN_PATTERN = re.compile(
    r"^(?P<family>.+?)_(?P<parameter>\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)$"
)


def ulp_distance(
    expected: npt.NDArray[np.float64], actual: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Count the representable doubles between two arrays of values.

    A distance of zero means the two values are the same double, one means they
    are neighbours, and a handful means the difference is floating point noise
    rather than a physical change.

    Parameters
    ----------
    expected : npt.NDArray[np.float64]
        Benchmark values.
    actual : npt.NDArray[np.float64]
        Freshly calculated values.

    Returns
    -------
    npt.NDArray[np.float64]
        The number of doubles between each pair, ``nan`` where either value is
        not finite. Counts are returned as floats so they can carry ``nan``.
    """

    def parts(
        values: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.int64]]:
        """Split doubles into their sign and their magnitude bit pattern.

        Parameters
        ----------
        values : npt.NDArray[np.float64]
            The values to split.

        Returns
        -------
        tuple[npt.NDArray[np.bool_], npt.NDArray[np.int64]]
            Whether each value is negative, and the integer ordering of its
            magnitude (monotonic in the value's magnitude).
        """
        bits = np.ascontiguousarray(values, dtype=np.float64).view(np.uint64)
        sign = np.uint64(1) << np.uint64(63)
        return (bits & sign) != 0, (bits & ~sign).astype(np.int64)

    expected = np.asarray(expected, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    finite = np.isfinite(expected) & np.isfinite(actual)
    expected_negative, expected_magnitude = parts(np.where(finite, expected, 0.0))
    actual_negative, actual_magnitude = parts(np.where(finite, actual, 0.0))
    # Counting across zero would overflow the signed integer axis, so the two
    # magnitudes are added there instead of subtracted.
    distance = np.where(
        expected_negative == actual_negative,
        np.abs(expected_magnitude - actual_magnitude).astype(np.float64),
        expected_magnitude.astype(np.float64) + actual_magnitude.astype(np.float64),
    )
    return np.where(finite, distance, np.nan)


def parse_column(column: str) -> tuple[str, float | None]:
    """Split an intensity measure column name into a family and a parameter.

    Parameters
    ----------
    column : str
        Column name, for example ``PGA``, ``pSA_0.05`` or ``FAS_0.0131826``.

    Returns
    -------
    tuple[str, float | None]
        The family name and the parameter value encoded in the column name, or
        ``None`` when the column carries no parameter.
    """
    match = _COLUMN_PATTERN.match(column)
    if match is None:
        return column, None
    return match["family"], float(match["parameter"])


@dataclass
class Comparison:
    """A cell-by-cell comparison of two intensity measure tables.

    Both frames are aligned onto the union of their rows and columns before
    anything is computed, so a missing intensity measure shows up as a missing
    value rather than an exception.
    """

    expected: pd.DataFrame
    """Benchmark values, indexed by component (or station), one column per
    intensity measure."""

    actual: pd.DataFrame
    """Freshly calculated values in the same layout."""

    atol: float = DEFAULT_ATOL
    """Absolute tolerance used to decide whether a value has changed."""

    rtol: float = DEFAULT_RTOL
    """Relative tolerance used to decide whether a value has changed."""

    @classmethod
    def from_frames(
        cls,
        expected: pd.DataFrame,
        actual: pd.DataFrame,
        atol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL,
    ) -> "Comparison":
        """Align two frames and wrap them in a comparison.

        Parameters
        ----------
        expected : pd.DataFrame
            Benchmark values.
        actual : pd.DataFrame
            Freshly calculated values.
        atol : float
            Absolute tolerance used to decide whether a value has changed.
        rtol : float
            Relative tolerance used to decide whether a value has changed.

        Returns
        -------
        Comparison
            The aligned comparison.
        """
        expected, actual = expected.align(actual, join="outer", axis=None)
        return cls(expected.astype(float), actual.astype(float), atol=atol, rtol=rtol)

    @cached_property
    def difference(self) -> pd.DataFrame:
        """Signed absolute difference, actual minus expected.

        Returns
        -------
        pd.DataFrame
            The difference of every cell.
        """
        return self.actual - self.expected

    @cached_property
    def relative(self) -> pd.DataFrame:
        """Signed difference relative to the magnitude of the benchmark.

        Returns
        -------
        pd.DataFrame
            The relative difference of every cell, ``nan`` where the benchmark
            value is zero or missing.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            relative = self.difference / self.expected.abs()
        return relative.replace([np.inf, -np.inf], np.nan)

    @cached_property
    def ulp(self) -> pd.DataFrame:
        """Distance between the two values counted in representable doubles.

        Returns
        -------
        pd.DataFrame
            The number of doubles between every pair of values, ``nan`` where
            either side is missing.
        """
        return pd.DataFrame(
            ulp_distance(
                self.expected.to_numpy(dtype=float), self.actual.to_numpy(dtype=float)
            ),
            index=self.expected.index,
            columns=self.expected.columns,
        )

    @cached_property
    def comparable(self) -> pd.DataFrame:
        """Cells where both frames hold a finite value.

        Returns
        -------
        pd.DataFrame
            Boolean mask of the comparable cells.
        """
        return np.isfinite(self.expected) & np.isfinite(self.actual)

    @cached_property
    def missing(self) -> pd.DataFrame:
        """Cells present in exactly one of the two frames.

        Returns
        -------
        pd.DataFrame
            Boolean mask of the cells missing on one side.
        """
        return self.expected.isna() ^ self.actual.isna()

    @cached_property
    def failing(self) -> pd.DataFrame:
        """Cells that fall outside the comparison tolerances.

        Returns
        -------
        pd.DataFrame
            Boolean mask of the failing cells.
        """
        close = np.isclose(
            self.actual.to_numpy(dtype=float),
            self.expected.to_numpy(dtype=float),
            atol=self.atol,
            rtol=self.rtol,
            equal_nan=True,
        )
        return pd.DataFrame(
            ~close, index=self.expected.index, columns=self.expected.columns
        )

    @cached_property
    def families(self) -> dict[str, list[str]]:
        """Column names grouped by intensity measure family.

        Returns
        -------
        dict[str, list[str]]
            Family name to the columns of that family, in column order.
        """
        families: dict[str, list[str]] = {}
        for column in self.expected.columns:
            family, _ = parse_column(str(column))
            families.setdefault(family, []).append(str(column))
        return families

    def parameters(self, columns: Sequence[str]) -> npt.NDArray[np.float64]:
        """Return the parameter (period, frequency, ...) of each column.

        Parameters
        ----------
        columns : Sequence[str]
            Column names belonging to a single parameterised family.

        Returns
        -------
        npt.NDArray[np.float64]
            The parameter of each column, ``nan`` where a column has none.
        """
        return np.array(
            [
                parameter
                if (parameter := parse_column(str(c))[1]) is not None
                else np.nan
                for c in columns
            ],
            dtype=np.float64,
        )

    def row_order(self) -> list[str]:
        """Order rows by how far they stray from the benchmark.

        Returns
        -------
        list[str]
            Row labels, worst first, used to pick which rows to plot when there
            are more rows than the palette can distinguish.
        """
        worst = self.relative.abs().max(axis=1).fillna(-1.0)
        return [str(label) for label in worst.sort_values(ascending=False).index]


@dataclass(frozen=True)
class Statistics:
    """Headline statistics over a set of compared cells."""

    compared: int
    """Number of cells where both frames hold a finite value."""

    failing: int
    """Number of cells outside tolerance."""

    missing: int
    """Number of cells present in only one of the two frames."""

    identical: int
    """Number of comparable cells that are bit-for-bit equal."""

    max_relative: float
    """Largest absolute relative difference."""

    max_relative_at: str
    """Label of the cell holding that largest relative difference."""

    median_relative: float
    """Median absolute relative difference."""

    p95_relative: float
    """95th percentile of the absolute relative difference."""

    max_absolute: float
    """Largest absolute difference."""

    max_absolute_at: str
    """Label of the cell holding that largest absolute difference."""

    median_ulp: float
    """Median distance in representable doubles."""

    noise: int
    """Number of comparable cells within `NOISE_ULP` doubles of the benchmark."""


def _locate(frame: pd.DataFrame) -> tuple[float, str]:
    """Find the largest value in a frame and where it sits.

    Parameters
    ----------
    frame : pd.DataFrame
        Frame of non-negative magnitudes.

    Returns
    -------
    tuple[float, str]
        The largest value and a ``column @ row`` label for it, or ``nan`` and an
        em dash when the frame holds no finite value.
    """
    array = frame.to_numpy(dtype=float)
    if array.size == 0 or not np.isfinite(array).any():
        return float("nan"), "—"
    flat = int(np.nanargmax(array))
    row, column = int(flat // array.shape[1]), int(flat % array.shape[1])
    return float(array[row, column]), f"{frame.columns[column]} @ {frame.index[row]}"


def summarise(
    comparison: Comparison, columns: Sequence[str] | None = None
) -> Statistics:
    """Compute headline statistics for (part of) a comparison.

    Parameters
    ----------
    comparison : Comparison
        The comparison to summarise.
    columns : Sequence[str] | None
        Restrict the statistics to these columns; all columns when ``None``.

    Returns
    -------
    Statistics
        The computed statistics.
    """
    columns = list(comparison.expected.columns) if columns is None else list(columns)
    comparable = comparison.comparable[columns]
    relative = comparison.relative[columns].where(comparable).abs()
    absolute = comparison.difference[columns].where(comparable).abs()
    finite_relative = relative.to_numpy(dtype=float)
    finite_relative = finite_relative[np.isfinite(finite_relative)]
    ulp = comparison.ulp[columns].where(comparable).to_numpy(dtype=float)
    finite_ulp = ulp[np.isfinite(ulp)]
    max_relative, max_relative_at = _locate(relative)
    max_absolute, max_absolute_at = _locate(absolute)
    return Statistics(
        compared=int(comparable.to_numpy().sum()),
        failing=int(comparison.failing[columns].to_numpy().sum()),
        missing=int(comparison.missing[columns].to_numpy().sum()),
        identical=int((absolute == 0).to_numpy().sum()),
        max_relative=max_relative,
        max_relative_at=max_relative_at,
        median_relative=float(np.median(finite_relative))
        if finite_relative.size
        else float("nan"),
        p95_relative=float(np.percentile(finite_relative, 95))
        if finite_relative.size
        else float("nan"),
        max_absolute=max_absolute,
        max_absolute_at=max_absolute_at,
        median_ulp=float(np.median(finite_ulp)) if finite_ulp.size else float("nan"),
        noise=int((finite_ulp <= NOISE_ULP).sum()),
    )


def format_value(value: float, digits: int = 4) -> str:
    """Format a number for an axis tick, table cell or tooltip.

    Parameters
    ----------
    value : float
        Value to format.
    digits : int
        Number of significant digits to keep.

    Returns
    -------
    str
        The formatted value, or an em dash when the value is not finite.
    """
    if value is None or not np.isfinite(value):
        return "—"
    if value == 0:
        return "0"
    text = f"{value:,.{digits}g}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        text = f"{mantissa}e{int(exponent)}"
    return text


def format_percent(value: float, digits: int = 3) -> str:
    """Format a fractional difference as a signed percentage.

    Parameters
    ----------
    value : float
        Difference expressed as a fraction of the benchmark value.
    digits : int
        Number of significant digits to keep.

    Returns
    -------
    str
        The formatted percentage, or an em dash when the value is not finite.
    """
    if value is None or not np.isfinite(value):
        return "—"
    percent = value * 100
    if percent == 0:
        return "0%"
    text = f"{percent:+.{digits}g}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        text = f"{mantissa}e{int(exponent)}"
    return f"{text}%"


def format_ulp(value: float) -> str:
    """Format a distance in representable doubles.

    Parameters
    ----------
    value : float
        The number of doubles between two values.

    Returns
    -------
    str
        The formatted count, or an em dash when the value is not finite.
    """
    if value is None or not np.isfinite(value):
        return "—"
    if value < 1e6:
        return f"{int(value):,}"
    return format_value(value, 3)


def _nice_step(span: float, target: int) -> float:
    """Round a tick spacing to the nearest 1, 2, 2.5 or 5 times a power of ten.

    Parameters
    ----------
    span : float
        Width of the axis domain.
    target : int
        Rough number of ticks wanted.

    Returns
    -------
    float
        The rounded tick spacing.
    """
    raw = span / max(target, 1)
    if raw <= 0 or not np.isfinite(raw):
        return 1.0
    magnitude = 10 ** math.floor(math.log10(raw))
    for multiple in (1, 2, 2.5, 5):
        if raw <= multiple * magnitude:
            return multiple * magnitude
    return 10 * magnitude


def _linear_ticks(low: float, high: float, target: int = 6) -> list[float]:
    """Generate evenly spaced ticks covering a linear domain.

    Parameters
    ----------
    low : float
        Lower bound of the domain.
    high : float
        Upper bound of the domain.
    target : int
        Rough number of ticks wanted.

    Returns
    -------
    list[float]
        The tick values inside the domain.
    """
    step = _nice_step(high - low, target)
    start = math.ceil(low / step) * step
    ticks = []
    value = start
    while value <= high + step * 1e-6:
        ticks.append(0.0 if abs(value) < step * 1e-6 else value)
        value += step
    return ticks


def _log_ticks(low: float, high: float, target: int = 6) -> list[float]:
    """Generate decade (and, over narrow ranges, sub-decade) ticks.

    Parameters
    ----------
    low : float
        Lower bound of the domain, strictly positive.
    high : float
        Upper bound of the domain.
    target : int
        Rough number of ticks wanted.

    Returns
    -------
    list[float]
        The tick values inside the domain.
    """
    first, last = math.floor(math.log10(low)), math.ceil(math.log10(high))
    decades = list(range(first, last + 1))
    multiples: tuple[float, ...] = (1.0,)
    if len(decades) <= 2:
        multiples = (1.0, 2.0, 3.0, 5.0)
    elif len(decades) <= 4:
        multiples = (1.0, 3.0)
    stride = max(1, len(decades) // max(target, 1))
    ticks = [
        multiple * 10.0**decade
        for decade in decades[::stride]
        for multiple in multiples
    ]
    return [tick for tick in ticks if low <= tick <= high]


class _Scale:
    """Maps data values onto pixel positions along one axis.

    Parameters
    ----------
    low : float
        Lower bound of the data domain.
    high : float
        Upper bound of the data domain.
    pixel_low : float
        Pixel position of ``low``.
    pixel_high : float
        Pixel position of ``high``.
    log : bool
        Whether the axis is logarithmic.
    """

    def __init__(
        self,
        low: float,
        high: float,
        pixel_low: float,
        pixel_high: float,
        log: bool = False,
    ) -> None:
        """Build a scale.

        Parameters
        ----------
        low : float
            Lower bound of the data domain.
        high : float
            Upper bound of the data domain.
        pixel_low : float
            Pixel position of ``low``.
        pixel_high : float
            Pixel position of ``high``.
        log : bool
            Whether the axis is logarithmic.
        """
        if not np.isfinite(low) or not np.isfinite(high):
            low, high = (0.0, 1.0)
        if log:
            low = max(low, 1e-30)
            if high <= low:
                high = low * 10
        if high <= low:
            padding = abs(low) * 0.1 or 1.0
            low, high = low - padding, high + padding
        self.low = low
        self.high = high
        self.pixel_low = pixel_low
        self.pixel_high = pixel_high
        self.log = log

    def __call__(self, value: float) -> float:
        """Convert a data value to a pixel position.

        Parameters
        ----------
        value : float
            The data value.

        Returns
        -------
        float
            The pixel position, ``nan`` when the value cannot be placed.
        """
        if value is None or not np.isfinite(value):
            return float("nan")
        if self.log:
            if value <= 0:
                return float("nan")
            fraction = (math.log10(value) - math.log10(self.low)) / (
                math.log10(self.high) - math.log10(self.low)
            )
        else:
            fraction = (value - self.low) / (self.high - self.low)
        return self.pixel_low + fraction * (self.pixel_high - self.pixel_low)

    def ticks(self, target: int = 6) -> list[float]:
        """Return tick values for this axis.

        Parameters
        ----------
        target : int
            Rough number of ticks wanted.

        Returns
        -------
        list[float]
            The tick values.
        """
        if self.log:
            return _log_ticks(self.low, self.high, target)
        return _linear_ticks(self.low, self.high, target)


def _domain(
    values: npt.NDArray[np.float64], log: bool = False, pad: float = 0.05
) -> tuple[float, float]:
    """Compute a padded domain covering the finite entries of an array.

    Parameters
    ----------
    values : npt.NDArray[np.float64]
        Values that must fit inside the domain.
    log : bool
        Whether the axis is logarithmic (padding is applied in log space and
        non-positive values are ignored).
    pad : float
        Fraction of the span to add at each end.

    Returns
    -------
    tuple[float, float]
        The lower and upper bound of the domain.
    """
    finite = values[np.isfinite(values)]
    if log:
        finite = finite[finite > 0]
    if finite.size == 0:
        return (1.0, 10.0) if log else (0.0, 1.0)
    low, high = float(finite.min()), float(finite.max())
    if log:
        span = math.log10(high) - math.log10(low)
        span = span or 1.0
        return 10 ** (math.log10(low) - span * pad), 10 ** (
            math.log10(high) + span * pad
        )
    span = high - low
    if span == 0:
        span = abs(high) or 1.0
    return low - span * pad, high + span * pad


@dataclass
class _Series:
    """One line (or one set of dots) inside a chart."""

    name: str
    """Series name, shown in the legend and the tooltip."""

    values: npt.NDArray[np.float64]
    """Y values, one per shared x position."""

    slot: int | None = None
    """Categorical palette slot, or ``None`` for the muted baseline colour."""

    labels: list[str] = field(default_factory=list)
    """Pre-formatted value labels used by the tooltip."""


@dataclass(frozen=True)
class _LegendItem:
    """One entry of a chart legend."""

    label: str
    """Text of the entry."""

    slot: int | None
    """Categorical palette slot, or ``None`` for the muted baseline colour."""

    kind: str = "line"
    """``line`` for a stroke key, ``dot`` for a marker, ``rect`` for a swatch."""

    series: int | None = None
    """Index of the series this entry switches on and off, if it switches one."""


def _slot_class(slot: int | None) -> str:
    """Return the CSS class carrying the colour of a palette slot.

    Parameters
    ----------
    slot : int | None
        Palette slot index, or ``None`` for the muted baseline colour.

    Returns
    -------
    str
        The CSS class name.
    """
    if slot is None:
        return "muted"
    return f"s{slot % MAX_PLOTTED_SERIES}"


def _escape(text: object) -> str:
    """HTML-escape a value for inclusion in markup.

    Parameters
    ----------
    text : object
        Value to escape; converted with ``str`` first.

    Returns
    -------
    str
        The escaped text.
    """
    return html.escape(str(text), quote=True)


def _path(xs: Sequence[float], ys: Sequence[float]) -> str:
    """Build an SVG path, breaking it wherever a point is missing.

    Parameters
    ----------
    xs : Sequence[float]
        Pixel x positions.
    ys : Sequence[float]
        Pixel y positions.

    Returns
    -------
    str
        The path data, empty when nothing can be drawn.
    """
    commands: list[str] = []
    pen_down = False
    for x, y in zip(xs, ys, strict=True):
        if not (np.isfinite(x) and np.isfinite(y)):
            pen_down = False
            continue
        commands.append(f"{'L' if pen_down else 'M'}{x:.2f} {y:.2f}")
        pen_down = True
    return " ".join(commands)


def _figure(
    title: str,
    note: str,
    svg: str,
    legend: Sequence[_LegendItem] = (),
    chart_data: Mapping[str, object] | None = None,
    table: str = "",
) -> str:
    """Wrap a chart in a figure with a caption, legend and optional table twin.

    Parameters
    ----------
    title : str
        Figure title.
    note : str
        Sub-title describing how to read the figure.
    svg : str
        The chart markup.
    legend : Sequence[_LegendItem]
        Legend entries; a legend is always drawn for two or more series.
    chart_data : Mapping[str, object] | None
        Geometry and labels used by the hover layer.
    table : str
        Markup for the collapsible table view of the same data.

    Returns
    -------
    str
        The figure markup.
    """
    parts = [
        '<figure class="chart">',
        '<figcaption><span class="chart-heading">'
        f'<span class="chart-title">{_escape(title)}</span>'
        + (f'<span class="chart-note">{_escape(note)}</span>' if note else "")
        + "</span>"
        f'<button type="button" class="chart-zoom" title="Enlarge this chart" '
        f'aria-label="Enlarge {_escape(title)}">⤢</button></figcaption>',
    ]
    if len(legend) > 1:
        entries = []
        for item in legend:
            key = f'<span class="key key-{item.kind} {_slot_class(item.slot)}"></span>'
            if item.series is None:
                entries.append(f"<li>{key}{_escape(item.label)}</li>")
            else:
                entries.append(
                    f'<li><button type="button" class="legend-toggle" '
                    f'data-series="{item.series}" aria-pressed="true" '
                    f'title="Show or hide {_escape(item.label)}">'
                    f"{key}{_escape(item.label)}</button></li>"
                )
        parts.append(f'<ul class="legend">{"".join(entries)}</ul>')
    parts.append(f'<div class="plot">{svg}</div>')
    if chart_data is not None:
        parts.append(
            '<script type="application/json" class="chart-data">'
            f"{json.dumps(chart_data)}</script>"
        )
    if table:
        parts.append(
            f'<details class="table-twin"><summary>Table view</summary>{table}</details>'
        )
    parts.append("</figure>")
    return "".join(parts)


def _axes(
    x_scale: _Scale,
    y_scale: _Scale,
    plot: tuple[float, float, float, float],
    x_label: str,
    y_label: str,
    x_format: str = "value",
    y_format: str = "value",
    x_ticks: int = 6,
    y_ticks: int = 5,
) -> str:
    """Draw grid lines, tick labels and axis titles.

    Parameters
    ----------
    x_scale : _Scale
        Scale of the horizontal axis.
    y_scale : _Scale
        Scale of the vertical axis.
    plot : tuple[float, float, float, float]
        Plot rectangle as ``(left, top, right, bottom)`` in pixels.
    x_label : str
        Title of the horizontal axis.
    y_label : str
        Title of the vertical axis.
    x_format : str
        ``value``, ``percent`` or ``magnitude`` formatting for the horizontal
        tick labels.
    y_format : str
        ``value``, ``percent`` or ``magnitude`` formatting for the vertical tick
        labels.
    x_ticks : int
        Rough number of horizontal ticks.
    y_ticks : int
        Rough number of vertical ticks.

    Returns
    -------
    str
        The axis markup.
    """
    left, top, right, bottom = plot
    formatters = {
        "value": format_value,
        "percent": format_percent,
        "magnitude": lambda tick: format_percent(tick).lstrip("+"),
    }
    x_formatter, y_formatter = formatters[x_format], formatters[y_format]
    parts = []
    for tick in y_scale.ticks(y_ticks):
        y = y_scale(tick)
        parts.append(
            f'<line class="grid" x1="{left}" x2="{right}" y1="{y:.2f}" y2="{y:.2f}"/>'
        )
        parts.append(
            f'<text class="tick tick-y" x="{left - 8}" y="{y + 3.5:.2f}">'
            f"{_escape(y_formatter(tick))}</text>"
        )
    for tick in x_scale.ticks(x_ticks):
        x = x_scale(tick)
        parts.append(
            f'<line class="grid" x1="{x:.2f}" x2="{x:.2f}" y1="{top}" y2="{bottom}"/>'
        )
        parts.append(
            f'<text class="tick tick-x" x="{x:.2f}" y="{bottom + 18}">'
            f"{_escape(x_formatter(tick))}</text>"
        )
    parts.append(
        f'<line class="axis" x1="{left}" x2="{right}" y1="{bottom}" y2="{bottom}"/>'
        f'<line class="axis" x1="{left}" x2="{left}" y1="{top}" y2="{bottom}"/>'
    )
    parts.append(
        f'<text class="axis-label" x="{(left + right) / 2:.1f}" y="{bottom + 38}">'
        f"{_escape(x_label)}</text>"
    )
    parts.append(
        f'<text class="axis-label" transform="rotate(-90 14 {(top + bottom) / 2:.1f})" '
        f'x="14" y="{(top + bottom) / 2:.1f}">{_escape(y_label)}</text>'
    )
    return "".join(parts)


def _tick_margin(scale: "_Scale", tick_format: str, ticks: int = 5) -> float:
    """Measure how much room the tick labels of a vertical axis need.

    Parameters
    ----------
    scale : _Scale
        The vertical scale whose ticks are drawn.
    tick_format : str
        ``value``, ``percent`` or ``magnitude`` tick formatting.
    ticks : int
        Rough number of ticks the axis will draw.

    Returns
    -------
    float
        Width in user units to reserve on the left of the plot.
    """
    formatters = {
        "value": format_value,
        "percent": format_percent,
        "magnitude": lambda tick: format_percent(tick).lstrip("+"),
    }
    formatter = formatters[tick_format]
    widest = max((len(formatter(tick)) for tick in scale.ticks(ticks)), default=4)
    return float(min(max(46.0, widest * 6.4 + 18.0), 116.0))


def _line_chart(
    x_values: npt.NDArray[np.float64],
    series: Sequence[_Series],
    *,
    title: str,
    note: str,
    x_label: str,
    y_label: str,
    x_log: bool = False,
    y_log: bool = False,
    y_format: str = "value",
    x_format: str = "value",
    bands: Sequence[tuple[float, float, str]] = (),
    zero_line: bool = False,
    y_domain: tuple[float, float] | None = None,
    width: int = 760,
    height: int = 340,
    legend: Sequence[_LegendItem] | None = None,
    table: str = "",
) -> str:
    """Draw a line chart of one or more series sharing an x axis.

    Parameters
    ----------
    x_values : npt.NDArray[np.float64]
        Shared x positions.
    series : Sequence[_Series]
        The series to draw.
    title : str
        Figure title.
    note : str
        Sub-title describing how to read the figure.
    x_label : str
        Title of the horizontal axis.
    y_label : str
        Title of the vertical axis.
    x_log : bool
        Whether the horizontal axis is logarithmic.
    y_log : bool
        Whether the vertical axis is logarithmic.
    y_format : str
        ``value`` or ``percent`` formatting for the vertical tick labels.
    x_format : str
        ``value`` or ``percent`` formatting for the horizontal tick labels.
    bands : Sequence[tuple[float, float, str]]
        Shaded horizontal bands given as ``(low, high, label)``.
    zero_line : bool
        Whether to draw an emphasised line at ``y = 0``.
    y_domain : tuple[float, float] | None
        Explicit vertical domain; derived from the data when ``None``.
    width : int
        Chart width in user units.
    height : int
        Chart height in user units.
    legend : Sequence[_LegendItem] | None
        Legend entries; derived from the series when ``None``.
    table : str
        Markup for the collapsible table view of the same data.

    Returns
    -------
    str
        The figure markup.
    """
    top, right, bottom = 16.0, float(width - 18), float(height - 46)
    if y_domain is None:
        stacked = np.concatenate([s.values for s in series]) if series else np.zeros(1)
        y_domain = _domain(stacked, log=y_log, pad=0.08)
    y_scale = _Scale(*y_domain, bottom, top, log=y_log)
    left = _tick_margin(y_scale, y_format)
    x_scale = _Scale(*_domain(x_values, log=x_log, pad=0.02), left, right, log=x_log)

    parts = [
        _axes(
            x_scale,
            y_scale,
            (left, top, right, bottom),
            x_label,
            y_label,
            x_format=x_format,
            y_format=y_format,
        )
    ]
    for low, high, _ in bands:
        y_high, y_low = y_scale(high), y_scale(low)
        if np.isfinite(y_high) and np.isfinite(y_low):
            parts.insert(
                0,
                f'<rect class="band" x="{left}" y="{min(y_high, y_low):.2f}" '
                f'width="{right - left:.2f}" height="{abs(y_low - y_high):.2f}"/>',
            )
    if zero_line:
        y = y_scale(0.0)
        if np.isfinite(y):
            parts.append(
                f'<line class="zero" x1="{left}" x2="{right}" y1="{y:.2f}" y2="{y:.2f}"/>'
            )

    x_pixels = [x_scale(value) for value in x_values]
    markers = len(x_values) <= 30
    clip_id = _next_clip_id()
    parts.append(
        f'<clipPath id="{clip_id}"><rect x="{left}" y="{top}" '
        f'width="{right - left:.2f}" height="{bottom - top:.2f}"/></clipPath>'
        f'<g clip-path="url(#{clip_id})">'
    )
    chart_series: list[dict[str, object]] = []
    for index, item in enumerate(series):
        y_pixels = [y_scale(value) for value in item.values]
        parts.append(f'<g class="series" data-series="{index}">')
        parts.append(
            f'<path class="line {_slot_class(item.slot)}" d="{_path(x_pixels, y_pixels)}"/>'
        )
        if markers:
            parts.extend(
                f'<circle class="dot {_slot_class(item.slot)}" cx="{x:.2f}" cy="{y:.2f}" r="3.5"/>'
                for x, y in zip(x_pixels, y_pixels, strict=True)
                if np.isfinite(x) and np.isfinite(y)
            )
        parts.append("</g>")
        chart_series.append(
            {
                "name": item.name,
                "slot": _slot_class(item.slot),
                "y": [None if not np.isfinite(y) else round(y, 2) for y in y_pixels],
                "labels": item.labels or [format_value(value) for value in item.values],
            }
        )

    parts.append("</g>")
    svg = (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_escape(title)}">'
        f"{''.join(parts)}</svg>"
    )
    chart_data = {
        "plot": [left, top, right, bottom],
        "x": [None if not np.isfinite(x) else round(x, 2) for x in x_pixels],
        "xLabels": [format_value(value) for value in x_values],
        "xLabel": x_label,
        "series": chart_series,
    }
    if legend is None:
        legend = [
            _LegendItem(item.name, item.slot, "line", index)
            for index, item in enumerate(series)
        ]
        legend += [_LegendItem(label, None, "rect") for *_, label in bands]
    return _figure(title, note, svg, legend, chart_data, table)


def _dot_plot(
    categories: Sequence[str],
    series: Sequence[_Series],
    *,
    title: str,
    note: str,
    x_label: str,
    bands: Sequence[tuple[float, float, str]] = (),
    width: int = 760,
    row_height: int = 26,
    table: str = "",
) -> str:
    """Draw one row of dots per category, one dot per series.

    Parameters
    ----------
    categories : Sequence[str]
        Row labels, drawn down the left hand side.
    series : Sequence[_Series]
        One entry per series, each holding one value per category.
    title : str
        Figure title.
    note : str
        Sub-title describing how to read the figure.
    x_label : str
        Title of the horizontal axis.
    bands : Sequence[tuple[float, float, str]]
        Shaded vertical bands given as ``(low, high, label)``.
    width : int
        Chart width in user units.
    row_height : int
        Vertical space given to each category.
    table : str
        Markup for the collapsible table view of the same data.

    Returns
    -------
    str
        The figure markup.
    """
    spacing = 7.0
    row_height = max(row_height, int(spacing * len(series) + 14))
    height = int(row_height * len(categories) + 76)
    left, top, right, bottom = 96.0, 16.0, float(width - 18), float(height - 46)
    values = np.concatenate([s.values for s in series]) if series else np.zeros(1)
    band_edges = np.array([edge for low, high, _ in bands for edge in (low, high)])
    low, high = _domain(np.concatenate([values, band_edges, np.zeros(1)]), pad=0.08)
    x_scale = _Scale(low, high, left, right)
    y_positions = [top + row_height * (index + 0.5) for index in range(len(categories))]

    parts = []
    for band_low, band_high, _ in bands:
        x_low, x_high = x_scale(band_low), x_scale(band_high)
        parts.append(
            f'<rect class="band" x="{min(x_low, x_high):.2f}" y="{top}" '
            f'width="{abs(x_high - x_low):.2f}" height="{bottom - top:.2f}"/>'
        )
    for tick in x_scale.ticks(6):
        x = x_scale(tick)
        parts.append(
            f'<line class="grid" x1="{x:.2f}" x2="{x:.2f}" y1="{top}" y2="{bottom}"/>'
        )
        parts.append(
            f'<text class="tick tick-x" x="{x:.2f}" y="{bottom + 18}">'
            f"{_escape(format_percent(tick))}</text>"
        )
    zero = x_scale(0.0)
    parts.append(
        f'<line class="zero" x1="{zero:.2f}" x2="{zero:.2f}" y1="{top}" y2="{bottom}"/>'
    )
    for index in range(1, len(categories)):
        y = top + row_height * index
        parts.append(
            f'<line class="grid" x1="{left}" x2="{right}" y1="{y:.2f}" y2="{y:.2f}"/>'
        )
    for label, y in zip(categories, y_positions, strict=True):
        parts.append(
            f'<text class="tick tick-y" x="{left - 10}" y="{y + 3.5:.2f}">{_escape(label)}</text>'
        )
    for offset, item in enumerate(series):
        parts.append(f'<g class="series" data-series="{offset}">')
        dodge = (offset - (len(series) - 1) / 2) * spacing
        for index, (value, row_y) in enumerate(
            zip(item.values, y_positions, strict=True)
        ):
            x, y = x_scale(value), row_y + dodge
            if not np.isfinite(x):
                continue
            label = item.labels[index] if item.labels else format_percent(value)
            tip = f"{item.name} · {categories[index]}: {label}"
            parts.append(
                f'<circle class="dot hit {_slot_class(item.slot)}" cx="{x:.2f}" cy="{y:.2f}" '
                f'r="4.5" tabindex="0" data-tip="{_escape(tip)}"/>'
            )
        parts.append("</g>")
    parts.append(
        f'<line class="axis" x1="{left}" x2="{right}" y1="{bottom}" y2="{bottom}"/>'
        f'<text class="axis-label" x="{(left + right) / 2:.1f}" y="{bottom + 38}">'
        f"{_escape(x_label)}</text>"
    )
    svg = (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_escape(title)}">'
        f"{''.join(parts)}</svg>"
    )
    legend = [
        _LegendItem(item.name, item.slot, "dot", index)
        for index, item in enumerate(series)
    ]
    legend += [_LegendItem(label, None, "rect") for *_, label in bands]
    return _figure(title, note, svg, legend, None, table)


def _bar_chart(
    edges: npt.NDArray[np.float64],
    counts: npt.NDArray[np.float64],
    *,
    title: str,
    note: str,
    x_label: str,
    y_label: str,
    x_format: str = "percent",
    width: int = 372,
    height: int = 260,
    table: str = "",
) -> str:
    """Draw a histogram from bin edges and counts.

    Parameters
    ----------
    edges : npt.NDArray[np.float64]
        Bin edges, one more than the number of counts.
    counts : npt.NDArray[np.float64]
        Number of values falling in each bin.
    title : str
        Figure title.
    note : str
        Sub-title describing how to read the figure.
    x_label : str
        Title of the horizontal axis.
    y_label : str
        Title of the vertical axis.
    x_format : str
        ``value`` or ``percent`` formatting for the horizontal tick labels.
    width : int
        Chart width in user units.
    height : int
        Chart height in user units.
    table : str
        Markup for the collapsible table view of the same data.

    Returns
    -------
    str
        The figure markup.
    """
    top, right, bottom = 16.0, float(width - 14), float(height - 46)
    y_scale = _Scale(0.0, float(counts.max() or 1) * 1.08, bottom, top)
    left = _tick_margin(y_scale, "value", 4)
    x_scale = _Scale(float(edges[0]), float(edges[-1]), left, right)
    parts = [
        _axes(
            x_scale,
            y_scale,
            (left, top, right, bottom),
            x_label,
            y_label,
            x_format=x_format,
            x_ticks=4,
            y_ticks=4,
        )
    ]
    for index, count in enumerate(counts):
        if count <= 0:
            continue
        x_start, x_end = x_scale(edges[index]), x_scale(edges[index + 1])
        y = y_scale(count)
        tip = (
            f"{format_percent(edges[index])} to {format_percent(edges[index + 1])}: "
            f"{int(count):,} values"
        )
        parts.append(
            f'<rect class="bar s0 hit" x="{x_start + 1:.2f}" y="{y:.2f}" '
            f'width="{max(x_end - x_start - 2, 0.8):.2f}" height="{bottom - y:.2f}" '
            f'rx="2" tabindex="0" data-tip="{_escape(tip)}"/>'
        )
    svg = (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_escape(title)}">'
        f"{''.join(parts)}</svg>"
    )
    return _figure(title, note, svg, (), None, table)


def _blocks(
    comparison: Comparison, columns: Sequence[str]
) -> dict[str, npt.NDArray[np.float64]]:
    """Extract the numeric blocks of a set of columns as plain arrays.

    Parameters
    ----------
    comparison : Comparison
        The comparison being reported.
    columns : Sequence[str]
        Columns to extract, in the order they should appear.

    Returns
    -------
    dict[str, npt.NDArray[np.float64]]
        The expected, actual, difference, relative, ulp, failing and missing
        blocks, each shaped ``(rows, columns)``.
    """
    selection = list(columns)
    return {
        "expected": comparison.expected[selection].to_numpy(dtype=float),
        "actual": comparison.actual[selection].to_numpy(dtype=float),
        "difference": comparison.difference[selection].to_numpy(dtype=float),
        "relative": comparison.relative[selection].to_numpy(dtype=float),
        "ulp": comparison.ulp[selection].to_numpy(dtype=float),
        "failing": comparison.failing[selection].to_numpy(dtype=bool),
        "missing": comparison.missing[selection].to_numpy(dtype=bool),
    }


def _row_positions(comparison: Comparison) -> dict[str, int]:
    """Map every row label onto its position in the compared frames.

    Parameters
    ----------
    comparison : Comparison
        The comparison being reported.

    Returns
    -------
    dict[str, int]
        Row label to row position.
    """
    return {str(label): index for index, label in enumerate(comparison.expected.index)}


def _cell_class(relative: float, absolute: float, failing: bool, missing: bool) -> str:
    """Pick the shading class of a table cell.

    Parameters
    ----------
    relative : float
        Signed relative difference of the cell.
    absolute : float
        Signed absolute difference of the cell.
    failing : bool
        Whether the cell falls outside tolerance.
    missing : bool
        Whether the value is present in only one of the two frames.

    Returns
    -------
    str
        Space separated CSS classes for the cell.
    """
    if missing:
        return "lvl-na"
    if not np.isfinite(relative):
        if not np.isfinite(absolute) or absolute == 0:
            return "lvl-0"
        magnitude = 3 if failing else 1
    elif abs(relative) >= _MAJOR_LEVEL:
        magnitude = 3
    elif abs(relative) >= _MINOR_LEVEL:
        magnitude = 2
    elif failing:
        magnitude = 1
    else:
        return "lvl-0"
    direction = "p" if (relative if np.isfinite(relative) else absolute) > 0 else "n"
    return f"lvl-{direction}{magnitude}{' fail' if failing else ''}"


def _cell_tip(
    column: str,
    row: str,
    expected: float,
    actual: float,
    absolute: float,
    relative: float,
    ulp: float,
) -> str:
    """Build the hover text of a table cell.

    Parameters
    ----------
    column : str
        Intensity measure name.
    row : str
        Row label (component or station).
    expected : float
        Benchmark value.
    actual : float
        Freshly calculated value.
    absolute : float
        Signed absolute difference.
    relative : float
        Signed relative difference.
    ulp : float
        Distance between the two values in representable doubles.

    Returns
    -------
    str
        The tooltip text, newline separated.
    """
    return (
        f"{column} · {row}\n"
        f"benchmark {format_value(expected, 6)}\n"
        f"current   {format_value(actual, 6)}\n"
        f"Δ {format_value(absolute, 4)} ({format_percent(relative)})\n"
        f"{format_ulp(ulp)} ULP apart"
    )


def _difference_table(
    comparison: Comparison, columns: Sequence[str], axes: FamilyAxes | None
) -> str:
    """Render one intensity measure family as a colour coded table.

    Parameters
    ----------
    comparison : Comparison
        The comparison being reported.
    columns : Sequence[str]
        Columns of the family, in the order they should appear as rows.
    axes : FamilyAxes | None
        Axis metadata when the family is parameterised, otherwise ``None``.

    Returns
    -------
    str
        The table markup.
    """
    rows = [str(label) for label in comparison.expected.index]
    blocks = _blocks(comparison, columns)
    header_label = (
        f"{axes.parameter_label} ({axes.parameter_unit})"
        if axes
        else "Intensity measure"
    )
    head = "".join(f'<th scope="col">{_escape(row)}</th>' for row in rows)
    body = []
    for position, column in enumerate(columns):
        _, parameter = parse_column(column)
        cells = []
        row_failing = False
        for index, row in enumerate(rows):
            expected = float(blocks["expected"][index, position])
            actual = float(blocks["actual"][index, position])
            absolute = float(blocks["difference"][index, position])
            relative = float(blocks["relative"][index, position])
            failing = bool(blocks["failing"][index, position])
            missing = bool(blocks["missing"][index, position])
            ulp = float(blocks["ulp"][index, position])
            row_failing |= failing
            if missing:
                texts = ("missing", "missing", "missing")
            else:
                texts = (
                    format_percent(relative)
                    if np.isfinite(relative)
                    else format_value(absolute, 3),
                    format_value(absolute, 3),
                    format_ulp(ulp),
                )
            tip = _cell_tip(column, row, expected, actual, absolute, relative, ulp)
            cells.append(
                f'<td class="{_cell_class(relative, absolute, failing, missing)}" tabindex="0" '
                f'data-relative="{_escape(texts[0])}" data-absolute="{_escape(texts[1])}" '
                f'data-ulp="{_escape(texts[2])}" data-tip="{_escape(tip)}">'
                f"{_escape(texts[0])}</td>"
            )
        label = format_value(parameter) if parameter is not None and axes else column
        sub = f'<span class="row-sub">{_escape(column)}</span>' if axes else ""
        body.append(
            f'<tr data-name="{_escape(column)}" data-fail="{int(row_failing)}">'
            f'<th scope="row"><span class="row-label">{_escape(label)}</span>{sub}</th>'
            f"{''.join(cells)}</tr>"
        )
    return (
        '<div class="table-wrap"><table class="diff">'
        f'<thead><tr><th scope="col">{_escape(header_label)}</th>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _plot_rows(comparison: Comparison) -> tuple[list[str], str]:
    """Choose which rows to draw and explain the choice.

    Parameters
    ----------
    comparison : Comparison
        The comparison being reported.

    Returns
    -------
    tuple[list[str], str]
        The row labels to plot and a note describing any rows left out.
    """
    rows = [str(label) for label in comparison.expected.index]
    if len(rows) <= MAX_PLOTTED_SERIES:
        return rows, ""
    ordered = comparison.row_order()[:MAX_PLOTTED_SERIES]
    kept = [row for row in rows if row in set(ordered)]
    note = (
        f"Showing the {len(kept)} of {len(rows)} rows that differ most; "
        "every row is in the tables below."
    )
    return kept, note


def _relative_series(
    comparison: Comparison,
    columns: Sequence[str],
    rows: Sequence[str],
    slots: Mapping[str, int],
) -> list[_Series]:
    """Build one relative-difference series per row.

    Parameters
    ----------
    comparison : Comparison
        The comparison being reported.
    columns : Sequence[str]
        Columns to read, in x order.
    rows : Sequence[str]
        Rows to turn into series.
    slots : Mapping[str, int]
        Palette slot of each row label.

    Returns
    -------
    list[_Series]
        One series per row.
    """
    relative = comparison.relative[list(columns)].to_numpy(dtype=float)
    positions = _row_positions(comparison)
    series = []
    for row in rows:
        values = relative[positions[row]]
        if not np.isfinite(values).any():
            continue
        series.append(
            _Series(
                name=row,
                values=values,
                slot=slots[row],
                labels=[format_percent(value) for value in values],
            )
        )
    return series


def _spectra_panels(
    comparison: Comparison,
    columns: Sequence[str],
    parameters: npt.NDArray[np.float64],
    axes: FamilyAxes,
    rows: Sequence[str],
) -> str:
    """Draw a benchmark-against-current panel for each row.

    Parameters
    ----------
    comparison : Comparison
        The comparison being reported.
    columns : Sequence[str]
        Columns of the family, in parameter order.
    parameters : npt.NDArray[np.float64]
        Parameter (period, frequency, ...) of each column.
    axes : FamilyAxes
        Axis metadata for the family.
    rows : Sequence[str]
        Rows to draw a panel for.

    Returns
    -------
    str
        Markup for the shared legend and the grid of panels.
    """
    blocks = _blocks(comparison, columns)
    positions = _row_positions(comparison)
    panels = []
    for row in rows:
        expected = blocks["expected"][positions[row]]
        actual = blocks["actual"][positions[row]]
        if not (np.isfinite(expected).any() or np.isfinite(actual).any()):
            continue
        panels.append(
            _line_chart(
                parameters,
                [
                    _Series(
                        "Benchmark", expected, None, [format_value(v) for v in expected]
                    ),
                    _Series("Current", actual, 0, [format_value(v) for v in actual]),
                ],
                title=str(row),
                note="",
                x_label=f"{axes.parameter_label} ({axes.parameter_unit})",
                y_label=axes.value_label,
                x_log=True,
                y_log=True,
                width=420,
                height=290,
                legend=[],
            )
        )
    legend = (
        '<ul class="legend legend-shared">'
        '<li><span class="key key-line muted"></span>Benchmark</li>'
        '<li><span class="key key-line s0"></span>Current</li></ul>'
    )
    return f'{legend}<div class="panel-grid">{"".join(panels)}</div>'


def _tolerance_band(comparison: Comparison) -> list[tuple[float, float, str]]:
    """Return the shaded band marking the relative tolerance.

    Parameters
    ----------
    comparison : Comparison
        The comparison being reported.

    Returns
    -------
    list[tuple[float, float, str]]
        A single band spanning plus and minus the relative tolerance.
    """
    return [
        (
            -comparison.rtol,
            comparison.rtol,
            f"±{comparison.rtol:.3g} relative tolerance",
        )
    ]


def _symmetric_domain(
    values: npt.NDArray[np.float64], floor: float
) -> tuple[tuple[float, float], int]:
    """Pick a symmetric vertical domain that outliers cannot flatten.

    Parameters
    ----------
    values : npt.NDArray[np.float64]
        The values to be plotted.
    floor : float
        Smallest half-height the domain may take.

    Returns
    -------
    tuple[tuple[float, float], int]
        The domain and the number of values falling outside it.
    """
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        return (-floor, floor), 0
    # A handful of large differences must not squash everything else into the
    # zero line, so the domain follows the bulk of the values and clips the
    # rest; the caller reports how many were clipped.
    limit = max(float(np.percentile(finite, 99)) * 1.35, floor * 1.5)
    limit = min(limit, float(finite.max()) * 1.08)
    limit = max(limit, floor * 1.5)
    return (-limit, limit), int((finite > limit).sum())


def _family_section(comparison: Comparison, family: str, columns: Sequence[str]) -> str:
    """Render the charts and table of one parameterised family.

    Parameters
    ----------
    comparison : Comparison
        The comparison being reported.
    family : str
        Family name, e.g. ``pSA``.
    columns : Sequence[str]
        Columns belonging to the family.

    Returns
    -------
    str
        The section markup.
    """
    axes = FAMILY_AXES.get(family, DEFAULT_FAMILY_AXES)
    parameters = comparison.parameters(columns)
    order = np.argsort(parameters)
    columns = [columns[index] for index in order]
    parameters = parameters[order]

    statistics = summarise(comparison, columns)
    rows, row_note = _plot_rows(comparison)
    slots = {row: index for index, row in enumerate(rows)}
    series = _relative_series(comparison, columns, rows, slots)
    plotted = (
        np.concatenate([item.values for item in series]) if series else np.zeros(1)
    )
    domain, outside = _symmetric_domain(plotted, comparison.rtol)
    notes = [
        (
            "Positive means the current run is larger than the benchmark. "
            f"{statistics.failing:,} of {statistics.compared:,} values fall outside tolerance."
        )
    ]
    if outside:
        notes.append(f"{outside:,} points sit outside the plotted range.")
    if row_note:
        notes.append(row_note)

    charts = _line_chart(
        parameters,
        series,
        title=f"{family} difference from benchmark",
        note=" ".join(notes),
        x_label=f"{axes.parameter_label} ({axes.parameter_unit})",
        y_label="Relative difference",
        x_log=True,
        y_format="percent",
        bands=_tolerance_band(comparison),
        zero_line=True,
        y_domain=domain,
    )
    panels = _spectra_panels(comparison, columns, parameters, axes, rows)
    table = _difference_table(comparison, columns, axes)
    return (
        f'<section class="card" id="family-{_escape(family)}">'
        f"<h2>{_escape(family)}</h2>"
        f"{_statistics_strip(statistics)}"
        f"{charts}"
        f"<h3>{_escape(family)} spectra, benchmark against current (log–log)</h3>"
        f"{panels}"
        f'<details class="values" open><summary>All {len(columns):,} {family} values '
        '<span class="row-count"></span></summary>'
        f"{table}</details>"
        "</section>"
    )


def _scalar_section(comparison: Comparison, columns: Sequence[str]) -> str:
    """Render the chart and table of the intensity measures without a parameter.

    Parameters
    ----------
    comparison : Comparison
        The comparison being reported.
    columns : Sequence[str]
        The scalar columns.

    Returns
    -------
    str
        The section markup.
    """
    statistics = summarise(comparison, columns)
    rows, row_note = _plot_rows(comparison)
    slots = {row: index for index, row in enumerate(rows)}
    relative = comparison.relative[list(columns)].to_numpy(dtype=float)
    positions = _row_positions(comparison)
    series = []
    for row in rows:
        values = relative[positions[row]]
        series.append(
            _Series(
                row, values, slots[row], [format_percent(value) for value in values]
            )
        )
    note = (
        "Each dot is one component; the shaded band is the tolerance. "
        f"{statistics.failing:,} of {statistics.compared:,} values fall outside it."
    )
    chart = _dot_plot(
        list(columns),
        series,
        title="Scalar intensity measures",
        note=f"{note} {row_note}".strip(),
        x_label="Relative difference from benchmark",
        bands=_tolerance_band(comparison),
    )
    table = _difference_table(comparison, columns, None)
    return (
        '<section class="card" id="family-scalar"><h2>Scalar intensity measures</h2>'
        f"{_statistics_strip(statistics)}{chart}"
        f'<details class="values" open><summary>All {len(columns):,} scalar values '
        '<span class="row-count"></span></summary>'
        f"{table}</details></section>"
    )


def _statistics_strip(statistics: Statistics) -> str:
    """Render a compact row of statistics for a section.

    Parameters
    ----------
    statistics : Statistics
        Statistics of the section.

    Returns
    -------
    str
        The markup of the strip.
    """
    entries = [
        ("Values", f"{statistics.compared:,}"),
        ("Outside tolerance", f"{statistics.failing:,}"),
        (
            "Max |Δ|",
            f"{format_percent(statistics.max_relative)} ({statistics.max_relative_at})",
        ),
        ("Median |Δ|", format_percent(statistics.median_relative)),
        ("95th percentile |Δ|", format_percent(statistics.p95_relative)),
        ("Median ULP", format_ulp(statistics.median_ulp)),
        ("Within noise", f"{statistics.noise:,} of {statistics.compared:,}"),
    ]
    items = "".join(
        f"<div><dt>{_escape(label)}</dt><dd>{_escape(value)}</dd></div>"
        for label, value in entries
    )
    return f'<dl class="strip">{items}</dl>'


def _stat_tile(label: str, value: str, note: str = "", status: str = "") -> str:
    """Render a single statistic tile.

    Parameters
    ----------
    label : str
        Name of the statistic.
    value : str
        The value itself.
    note : str
        Supporting detail shown underneath.
    status : str
        ``good`` or ``critical`` to colour the value, empty for neutral.

    Returns
    -------
    str
        The tile markup.
    """
    status_class = f" status-{status}" if status else ""
    note_markup = f'<p class="tile-note">{_escape(note)}</p>' if note else ""
    return (
        f'<div class="tile"><p class="tile-label">{_escape(label)}</p>'
        f'<p class="tile-value{status_class}">{_escape(value)}</p>{note_markup}</div>'
    )


def _summary_section(
    comparison: Comparison, statistics: Statistics, metadata: Mapping[str, str]
) -> str:
    """Render the headline summary of the whole comparison.

    Parameters
    ----------
    comparison : Comparison
        The comparison being reported.
    statistics : Statistics
        Statistics over every compared value.
    metadata : Mapping[str, str]
        Extra key/value pairs describing the run.

    Returns
    -------
    str
        The section markup.
    """
    passed = statistics.failing == 0
    hero_status = "good" if passed else "critical"
    hero_note = (
        "Every value is within tolerance."
        if passed
        else f"of {statistics.compared:,} compared values are outside tolerance"
    )
    tiles = [
        _stat_tile(
            "Largest relative difference",
            format_percent(statistics.max_relative),
            statistics.max_relative_at,
        ),
        _stat_tile(
            "Median relative difference", format_percent(statistics.median_relative)
        ),
        _stat_tile(
            "95th percentile relative difference",
            format_percent(statistics.p95_relative),
        ),
        _stat_tile(
            "Largest absolute difference",
            format_value(statistics.max_absolute),
            statistics.max_absolute_at,
        ),
        _stat_tile(
            "Median distance in ULP",
            format_ulp(statistics.median_ulp),
            f"{statistics.noise:,} values within {NOISE_ULP:.0f} ULP "
            "(floating point noise)",
        ),
        _stat_tile(
            "Identical values",
            f"{statistics.identical:,}",
            f"{statistics.identical / statistics.compared:.1%} of compared values"
            if statistics.compared
            else "",
        ),
        _stat_tile(
            "Missing on one side",
            f"{statistics.missing:,}",
            "values present in only one of the two tables",
        ),
    ]
    metadata_markup = "".join(
        f"<div><dt>{_escape(key)}</dt><dd>{_escape(value)}</dd></div>"
        for key, value in metadata.items()
    )
    return (
        '<section class="card summary"><div class="hero">'
        f'<p class="hero-value status-{hero_status}">{statistics.failing:,}</p>'
        f'<p class="hero-label">values outside tolerance</p>'
        f'<p class="hero-note">{_escape(hero_note)}</p></div>'
        f'<div class="tiles">{"".join(tiles)}</div>'
        f'<dl class="meta">{metadata_markup}</dl>'
        "</section>"
    )


def _family_summary_table(comparison: Comparison) -> str:
    """Summarise every family in one table.

    Parameters
    ----------
    comparison : Comparison
        The comparison being reported.

    Returns
    -------
    str
        The table markup.
    """
    rows = []
    for family, columns in comparison.families.items():
        statistics = summarise(comparison, columns)
        status = "" if statistics.failing == 0 else ' class="row-fail"'
        rows.append(
            f'<tr{status}><th scope="row">{_escape(family)}</th>'
            f"<td>{len(columns):,}</td><td>{statistics.compared:,}</td>"
            f"<td>{statistics.failing:,}</td>"
            f"<td>{_escape(format_percent(statistics.max_relative))}</td>"
            f"<td>{_escape(format_percent(statistics.median_relative))}</td>"
            f"<td>{_escape(format_value(statistics.max_absolute))}</td>"
            f"<td>{_escape(format_ulp(statistics.median_ulp))}</td>"
            f'<td class="where">{_escape(statistics.max_relative_at)}</td></tr>'
        )
    return (
        '<div class="table-wrap"><table class="summary-table">'
        "<thead><tr><th>Family</th><th>Columns</th><th>Values</th><th>Outside tolerance</th>"
        "<th>Max |Δ| relative</th><th>Median |Δ| relative</th><th>Max |Δ| absolute</th>"
        "<th>Median ULP</th><th>Worst value</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _worst_table(comparison: Comparison, limit: int = 15) -> str:
    """List the values that differ most from the benchmark.

    Parameters
    ----------
    comparison : Comparison
        The comparison being reported.
    limit : int
        Maximum number of values to list.

    Returns
    -------
    str
        The table markup.
    """
    magnitude = (
        comparison.relative.abs().where(comparison.comparable).to_numpy(dtype=float)
    )
    failing = comparison.failing.to_numpy()
    interesting = failing | (np.nan_to_num(magnitude) > 0)
    ranked = sorted(
        (tuple(position) for position in np.argwhere(interesting)),
        key=lambda position: (
            not failing[position],
            -(magnitude[position] if np.isfinite(magnitude[position]) else -1.0),
        ),
    )
    expected = comparison.expected.to_numpy(dtype=float)
    actual = comparison.actual.to_numpy(dtype=float)
    difference = comparison.difference.to_numpy(dtype=float)
    relative = comparison.relative.to_numpy(dtype=float)
    labels = [str(label) for label in comparison.expected.index]
    names = [str(name) for name in comparison.expected.columns]
    rows = []
    for position in ranked[:limit]:
        classes = ' class="row-fail"' if failing[position] else ""
        flag = (
            '<span class="flag" title="outside tolerance">!</span>'
            if failing[position]
            else ""
        )
        rows.append(
            f'<tr{classes}><th scope="row">{_escape(names[position[1]])}</th>'
            f"<td>{_escape(labels[position[0]])}</td>"
            f"<td>{_escape(format_value(expected[position], 6))}</td>"
            f"<td>{_escape(format_value(actual[position], 6))}</td>"
            f"<td>{_escape(format_value(difference[position]))}</td>"
            f"<td>{_escape(format_percent(relative[position]))}{flag}</td></tr>"
        )
    if not rows:
        return '<p class="empty">Every value matches the benchmark exactly.</p>'
    return (
        '<div class="table-wrap"><table class="summary-table">'
        "<thead><tr><th>Intensity measure</th><th>Row</th><th>Benchmark</th><th>Current</th>"
        "<th>Δ</th><th>Δ relative</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _simple_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a plain table, used as the accessible twin of a chart.

    Parameters
    ----------
    headers : Sequence[str]
        Column headings.
    rows : Sequence[Sequence[str]]
        Table body, one sequence of cells per row.

    Returns
    -------
    str
        The table markup.
    """
    head = "".join(f"<th>{_escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_escape(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return (
        '<div class="table-wrap"><table class="summary-table">'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _distribution_section(comparison: Comparison) -> str:
    """Render the distribution of the differences across every compared value.

    Parameters
    ----------
    comparison : Comparison
        The comparison being reported.

    Returns
    -------
    str
        The section markup.
    """
    values = (
        comparison.relative.where(comparison.comparable).to_numpy(dtype=float).ravel()
    )
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return ""
    magnitude = np.abs(finite)
    limit = max(float(np.percentile(magnitude, 99.5)), comparison.rtol * 2)
    within = finite[magnitude <= limit]
    counts, edges = np.histogram(within, bins=41, range=(-limit, limit))
    clipped = int(finite.size - within.size)
    histogram = _bar_chart(
        edges,
        counts.astype(float),
        title="Distribution of differences",
        note=(
            f"{finite.size:,} comparable values"
            + (f", {clipped:,} beyond the plotted range" if clipped else "")
            + "."
        ),
        x_label="Relative difference",
        y_label="Values",
        table=_simple_table(
            ["From", "To", "Values"],
            [
                (
                    format_percent(edges[index]),
                    format_percent(edges[index + 1]),
                    f"{int(count):,}",
                )
                for index, count in enumerate(counts)
                if count
            ],
        ),
    )

    percentiles = np.linspace(0.0, 100.0, 101)
    magnitudes = np.percentile(magnitude, percentiles)
    positive = magnitudes[magnitudes > 0]
    low = float(positive.min()) / 2 if positive.size else comparison.rtol / 10
    high = max(float(magnitudes.max()), comparison.rtol * 2) * 1.4
    exact = int((magnitude == 0).sum())
    curve = _line_chart(
        percentiles,
        [
            _Series(
                "Relative difference",
                magnitudes,
                0,
                [format_percent(value) for value in magnitudes],
            )
        ],
        title="Error percentiles",
        note=(
            "Share of values (left to right) at or below a given difference. "
            f"{exact:,} values match the benchmark exactly."
        ),
        x_label="Percentile of compared values",
        y_label="|Relative difference|",
        y_log=True,
        y_format="magnitude",
        y_domain=(low, high),
        bands=[
            (low, comparison.rtol, f"within {comparison.rtol:.3g} relative tolerance")
        ],
        width=372,
        height=260,
        table=_simple_table(
            ["Percentile", "|Relative difference|"],
            [
                (f"{percentile:g}", format_percent(value))
                for percentile, value in zip(percentiles, magnitudes, strict=True)
                if percentile % 10 == 0 or percentile in (95.0, 99.0)
            ],
        ),
    )
    return (
        '<section class="card"><h2>Difference distribution</h2>'
        f'<div class="chart-row">{histogram}{curve}</div></section>'
    )


_STYLESHEET = """
:root {
  color-scheme: light;
  --page: #f9f9f7;
  --surface-1: #fcfcfb;
  --surface-2: #f3f2ee;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --text-muted: #898781;
  --grid: #e1e0d9;
  --axis: #c3c2b7;
  --border: rgba(11, 11, 11, 0.1);
  --band: rgba(11, 11, 11, 0.06);
  --good: #0ca30c;
  --critical: #d03b3b;
  --series-1: #2a78d6;
  --series-2: #eb6834;
  --series-3: #1baf7a;
  --series-4: #eda100;
  --series-5: #e87ba4;
  --series-6: #008300;
  --series-7: #4a3aa7;
  --series-8: #e34948;
  --pos-1: #fbe3e2;
  --pos-2: #f0a3a1;
  --pos-3: #d03b3b;
  --neg-1: #dfeafb;
  --neg-2: #9ec5f4;
  --neg-3: #2a78d6;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d;
    --surface-1: #1a1a19;
    --surface-2: #232321;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted: #898781;
    --grid: #2c2c2a;
    --axis: #383835;
    --border: rgba(255, 255, 255, 0.12);
    --band: rgba(255, 255, 255, 0.07);
    --series-1: #3987e5;
    --series-2: #d95926;
    --series-3: #199e70;
    --series-4: #c98500;
    --series-5: #d55181;
    --series-6: #008300;
    --series-7: #9085e9;
    --series-8: #e66767;
    --pos-1: #3a2120;
    --pos-2: #7a3130;
    --pos-3: #e34948;
    --neg-1: #152a45;
    --neg-2: #1c5cab;
    --neg-3: #3987e5;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d;
  --surface-1: #1a1a19;
  --surface-2: #232321;
  --text-primary: #ffffff;
  --text-secondary: #c3c2b7;
  --text-muted: #898781;
  --grid: #2c2c2a;
  --axis: #383835;
  --border: rgba(255, 255, 255, 0.12);
  --band: rgba(255, 255, 255, 0.07);
  --series-1: #3987e5;
  --series-2: #d95926;
  --series-3: #199e70;
  --series-4: #c98500;
  --series-5: #d55181;
  --series-6: #008300;
  --series-7: #9085e9;
  --series-8: #e66767;
  --pos-1: #3a2120;
  --pos-2: #7a3130;
  --pos-3: #e34948;
  --neg-1: #152a45;
  --neg-2: #1c5cab;
  --neg-3: #3987e5;
}

* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 0 20px 64px;
  background: var(--page);
  color: var(--text-primary);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
}
main, header.page-head { max-width: 1180px; margin: 0 auto; }
header.page-head { padding: 32px 0 8px; display: flex; gap: 16px; align-items: flex-start; }
header.page-head h1 { font-size: 24px; margin: 0 0 4px; letter-spacing: -0.01em; }
header.page-head p { margin: 0; color: var(--text-secondary); }
.head-text { flex: 1; }
button.theme {
  border: 1px solid var(--border); background: var(--surface-1); color: var(--text-secondary);
  border-radius: 8px; padding: 6px 12px; font: inherit; cursor: pointer;
}
button.theme:hover { color: var(--text-primary); }

.card {
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px 22px;
  margin: 16px 0;
}
.card h2 { font-size: 17px; margin: 0 0 14px; }
.card h3 { font-size: 14px; margin: 20px 0 8px; color: var(--text-secondary); font-weight: 600; }

.summary { display: grid; grid-template-columns: minmax(180px, 220px) 1fr; gap: 24px; align-items: start; }
.hero { border-right: 1px solid var(--border); padding-right: 20px; }
.hero-value { font-size: 52px; line-height: 1.05; margin: 0; font-weight: 600; }
.hero-label { margin: 4px 0 0; color: var(--text-secondary); }
.hero-note { margin: 8px 0 0; color: var(--text-muted); font-size: 13px; }
.status-good { color: var(--good); }
.status-critical { color: var(--critical); }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 14px 20px; }
.tile-label { margin: 0; color: var(--text-secondary); font-size: 12px; }
.tile-value { margin: 2px 0 0; font-size: 22px; font-weight: 600; }
.tile-note { margin: 2px 0 0; color: var(--text-muted); font-size: 12px; }
dl.meta { grid-column: 1 / -1; display: flex; flex-wrap: wrap; gap: 8px 10px; margin: 4px 0 0; }
dl.meta div {
  display: flex; gap: 6px; background: var(--surface-2); border-radius: 999px;
  padding: 3px 12px; font-size: 12px;
}
dl.meta dt { color: var(--text-muted); margin: 0; }
dl.meta dd { margin: 0; color: var(--text-secondary); font-variant-numeric: tabular-nums; }

dl.strip { display: flex; flex-wrap: wrap; gap: 6px 28px; margin: 0 0 16px; }
dl.strip div { display: flex; gap: 8px; align-items: baseline; }
dl.strip dt { color: var(--text-muted); font-size: 12px; margin: 0; }
dl.strip dd { margin: 0; font-size: 13px; font-variant-numeric: tabular-nums; }

.controls {
  display: flex; flex-wrap: wrap; gap: 12px 20px; align-items: center;
  position: sticky; top: 0; z-index: 5;
}
.controls label { display: flex; gap: 8px; align-items: center; color: var(--text-secondary); }
fieldset.metric {
  display: flex; gap: 4px 14px; align-items: center; border: 1px solid var(--border);
  border-radius: 8px; padding: 4px 12px 6px; margin: 0;
}
fieldset.metric legend { color: var(--text-muted); font-size: 12px; padding: 0 4px; }
.swatch-note { color: var(--text-muted); font-size: 12px; }
.controls input[type="search"] {
  border: 1px solid var(--border); background: var(--surface-2); color: inherit;
  border-radius: 8px; padding: 6px 10px; font: inherit; min-width: 220px;
}
.swatches { display: flex; gap: 10px; align-items: center; margin-left: auto; flex-wrap: wrap; }
.swatches span { display: flex; gap: 5px; align-items: center; color: var(--text-muted); font-size: 12px; }
.swatches i { width: 14px; height: 14px; border-radius: 3px; display: inline-block; }

figure.chart { margin: 0 0 8px; }
figure.chart figcaption { display: flex; gap: 10px; align-items: baseline; margin-bottom: 6px; }
.chart-heading { display: flex; flex-wrap: wrap; gap: 4px 10px; align-items: baseline; flex: 1; }
.chart-zoom, .zoom-controls button {
  flex: none; border: 1px solid var(--border); background: var(--surface-2);
  color: var(--text-secondary); border-radius: 6px; padding: 1px 8px 3px;
  font: inherit; line-height: 1.3; cursor: pointer;
}
.chart-zoom:hover, .zoom-controls button:hover { color: var(--text-primary); }
.chart-title { font-weight: 600; }
.chart-note { color: var(--text-muted); font-size: 12px; }
.plot svg { width: 100%; height: auto; display: block; overflow: visible; touch-action: none; }
.chart-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 24px; }
.panel-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 20px 26px; }
.panel-grid .chart-title { font-weight: 500; color: var(--text-secondary); }

ul.legend { display: flex; flex-wrap: wrap; gap: 4px 16px; list-style: none; margin: 0 0 8px; padding: 0; }
ul.legend li { display: flex; gap: 6px; align-items: center; color: var(--text-secondary); font-size: 12px; }
button.legend-toggle {
  display: flex; gap: 6px; align-items: center; border: none; background: none;
  color: inherit; font: inherit; padding: 2px 2px; cursor: pointer; border-radius: 4px;
}
button.legend-toggle:hover { color: var(--text-primary); background: var(--surface-2); }
button.legend-toggle[aria-pressed="false"] { opacity: 0.45; text-decoration: line-through; }
g.series[hidden] { display: none; }
.key { display: inline-block; }
.key-line { width: 16px; height: 2px; border-radius: 2px; background: var(--series); }
.key-dot { width: 9px; height: 9px; border-radius: 50%; background: var(--series); }
.key-rect { width: 14px; height: 10px; border-radius: 2px; background: var(--band); border: 1px solid var(--border); }

.s0 { --series: var(--series-1); }
.s1 { --series: var(--series-2); }
.s2 { --series: var(--series-3); }
.s3 { --series: var(--series-4); }
.s4 { --series: var(--series-5); }
.s5 { --series: var(--series-6); }
.s6 { --series: var(--series-7); }
.s7 { --series: var(--series-8); }
.muted { --series: var(--axis); }

path.line { fill: none; stroke: var(--series); stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
path.line.muted { stroke-width: 3.4; opacity: 1; }
circle.dot { fill: var(--series); stroke: var(--surface-1); stroke-width: 2; }
circle.hit { cursor: crosshair; }
circle.hit:focus { outline: none; stroke: var(--text-primary); }
rect.bar { fill: var(--series); }
rect.bar:hover, rect.bar:focus { fill-opacity: 0.75; outline: none; }
line.grid { stroke: var(--grid); stroke-width: 1; }
line.axis { stroke: var(--axis); stroke-width: 1; }
line.zero { stroke: var(--axis); stroke-width: 1.5; }
rect.band { fill: var(--band); }
text.tick { fill: var(--text-muted); font-size: 11px; font-variant-numeric: tabular-nums; }
text.tick-y { text-anchor: end; }
text.tick-x { text-anchor: middle; }
text.axis-label { fill: var(--text-secondary); font-size: 12px; text-anchor: middle; }
line.crosshair { stroke: var(--axis); stroke-width: 1; }
circle.hover-dot { fill: var(--series); stroke: var(--surface-1); stroke-width: 2; }
.hover-layer[hidden] { display: none; }

details { margin-top: 10px; }
details > summary {
  cursor: pointer; color: var(--text-secondary); font-size: 13px; padding: 6px 0;
  list-style-position: outside;
}
details > summary:hover { color: var(--text-primary); }
.row-count { color: var(--text-muted); font-size: 12px; margin-left: 4px; }
.row-count:not(:empty)::before { content: "· "; }

.table-wrap {
  overflow: auto; max-height: 70vh; border: 1px solid var(--border); border-radius: 10px;
  margin-top: 8px;
}
table { border-collapse: separate; border-spacing: 0; width: 100%; font-size: 12.5px; }
table th, table td { padding: 5px 10px; text-align: right; white-space: nowrap; }
table thead th {
  position: sticky; top: 0; z-index: 2; background: var(--surface-2);
  color: var(--text-secondary); font-weight: 600; text-align: right;
  border-bottom: 1px solid var(--border);
}
table tbody th {
  position: sticky; left: 0; z-index: 1; background: var(--surface-1); text-align: left;
  font-weight: 500; border-right: 1px solid var(--border);
}
table thead th:first-child { left: 0; z-index: 3; text-align: left; }
table tbody tr:nth-child(even) th { background: var(--surface-2); }
.summary-table tbody tr:nth-child(even) td { background: var(--surface-2); }
table.diff td { font-variant-numeric: tabular-nums; }
.row-sub { color: var(--text-muted); margin-left: 8px; font-size: 11px; }
.summary-table td, .summary-table th { font-variant-numeric: tabular-nums; }
.summary-table .where { text-align: left; color: var(--text-secondary); }
tr.row-fail th { color: var(--critical); }
.empty { color: var(--text-muted); }

td.lvl-0 { color: var(--text-muted); }
td.lvl-na { color: var(--text-muted); font-style: italic; }
td.lvl-p1 { background: var(--pos-1); }
td.lvl-p2 { background: var(--pos-2); }
td.lvl-p3 { background: var(--pos-3); color: #ffffff; }
td.lvl-n1 { background: var(--neg-1); }
td.lvl-n2 { background: var(--neg-2); }
td.lvl-n3 { background: var(--neg-3); color: #ffffff; }
:root[data-theme="dark"] td.lvl-p1, :root[data-theme="dark"] td.lvl-n1,
:root[data-theme="dark"] td.lvl-p2, :root[data-theme="dark"] td.lvl-n2 { color: var(--text-primary); }
td.fail { font-weight: 600; }
table.diff td.fail::after { content: " !"; font-weight: 700; }
.flag { margin-left: 4px; font-weight: 700; }
td:focus { outline: 2px solid var(--text-primary); outline-offset: -2px; }

dialog.zoom {
  width: 96vw; max-width: 96vw; height: 92vh; max-height: 92vh; padding: 0;
  border: 1px solid var(--border); border-radius: 12px;
  background: var(--surface-1); color: var(--text-primary);
}
dialog.zoom::backdrop { background: rgba(0, 0, 0, 0.55); }
.zoom-head {
  display: flex; gap: 12px; align-items: center; padding: 12px 16px;
  border-bottom: 1px solid var(--border);
}
.zoom-title { font-weight: 600; }
.zoom-controls { margin-left: auto; display: flex; gap: 6px; align-items: center; }
.zoom-level { color: var(--text-muted); font-size: 12px; font-variant-numeric: tabular-nums; }
.zoom-body { overflow: auto; height: calc(92vh - 58px); padding: 14px 16px 24px; }
.zoom-body > figure.chart { margin: 0; }
dialog.zoom .chart-zoom, dialog.zoom .chart-title { display: none; }
.zoom-body:focus { outline: none; }
dialog.zoom .chart-note { font-size: 13px; }

.tooltip {
  position: fixed; z-index: 50; pointer-events: none; white-space: pre-line;
  background: var(--surface-1); color: var(--text-primary);
  border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px;
  font-size: 12px; box-shadow: 0 6px 24px rgba(0, 0, 0, 0.18); max-width: 320px;
  font-variant-numeric: tabular-nums;
}
.tooltip[hidden] { display: none; }
.tooltip .tip-row { display: flex; gap: 8px; align-items: center; }
.tooltip .tip-row .key-line { flex: none; }
.tooltip .tip-name { color: var(--text-secondary); }
.tooltip .tip-value { margin-left: auto; font-weight: 600; }
.tooltip .tip-head { color: var(--text-secondary); margin-bottom: 4px; }

@media (max-width: 720px) {
  .summary { grid-template-columns: 1fr; }
  .hero { border-right: none; border-bottom: 1px solid var(--border); padding: 0 0 12px; }
}
"""

_SCRIPT = """
(function () {
  var tooltip = document.createElement('div');
  tooltip.className = 'tooltip';
  tooltip.hidden = true;
  document.body.appendChild(tooltip);

  function place(x, y) {
    var pad = 14;
    var box = tooltip.getBoundingClientRect();
    var left = Math.min(x + pad, window.innerWidth - box.width - 8);
    var top = y - box.height - pad;
    if (top < 8) { top = y + pad; }
    tooltip.style.left = Math.max(8, left) + 'px';
    tooltip.style.top = top + 'px';
  }

  function showText(text, x, y) {
    tooltip.textContent = text;
    tooltip.hidden = false;
    place(x, y);
  }

  function showRows(head, rows, x, y) {
    tooltip.textContent = '';
    var header = document.createElement('div');
    header.className = 'tip-head';
    header.textContent = head;
    tooltip.appendChild(header);
    rows.forEach(function (row) {
      var line = document.createElement('div');
      line.className = 'tip-row';
      var key = document.createElement('span');
      key.className = 'key key-line ' + row.slot;
      var name = document.createElement('span');
      name.className = 'tip-name';
      name.textContent = row.name;
      var value = document.createElement('span');
      value.className = 'tip-value';
      value.textContent = row.value;
      line.appendChild(key);
      line.appendChild(name);
      line.appendChild(value);
      tooltip.appendChild(line);
    });
    tooltip.hidden = false;
    place(x, y);
  }

  function hide() { tooltip.hidden = true; }

  function target(event) {
    var node = event.target;
    return node && node.closest ? node.closest('[data-tip]') : null;
  }

  document.addEventListener('pointerover', function (event) {
    var node = target(event);
    if (node) { showText(node.getAttribute('data-tip'), event.clientX, event.clientY); }
  });
  document.addEventListener('pointermove', function (event) {
    if (!tooltip.hidden && target(event)) { place(event.clientX, event.clientY); }
  });
  document.addEventListener('pointerout', function (event) {
    if (target(event)) { hide(); }
  });
  document.addEventListener('focusin', function (event) {
    var node = target(event);
    if (node) {
      var box = node.getBoundingClientRect();
      showText(node.getAttribute('data-tip'), box.left + box.width / 2, box.top);
    }
  });
  document.addEventListener('focusout', hide);
  document.addEventListener('keydown', function (event) {
    if (event.key === 'Escape') { hide(); }
  });

  var NS = 'http://www.w3.org/2000/svg';
  document.querySelectorAll('figure.chart').forEach(function (figure) {
    var payload = figure.querySelector('script.chart-data');
    var svg = figure.querySelector('svg');
    if (!payload || !svg) { return; }
    var data = JSON.parse(payload.textContent);
    var plot = data.plot;
    var layer = document.createElementNS(NS, 'g');
    layer.setAttribute('class', 'hover-layer');
    layer.setAttribute('hidden', 'hidden');
    svg.appendChild(layer);
    var crosshair = document.createElementNS(NS, 'line');
    crosshair.setAttribute('class', 'crosshair');
    crosshair.setAttribute('y1', plot[1]);
    crosshair.setAttribute('y2', plot[3]);
    layer.appendChild(crosshair);
    var dots = data.series.map(function (series) {
      var dot = document.createElementNS(NS, 'circle');
      dot.setAttribute('class', 'hover-dot ' + series.slot);
      dot.setAttribute('r', '4');
      layer.appendChild(dot);
      return dot;
    });
    var groups = data.series.map(function (series, index) {
      return svg.querySelector('g.series[data-series="' + index + '"]');
    });

    function nearest(event) {
      var box = svg.getBoundingClientRect();
      var view = svg.viewBox.baseVal;
      var x = (event.clientX - box.left) / box.width * view.width;
      var best = -1;
      var distance = Infinity;
      for (var index = 0; index < data.x.length; index += 1) {
        var position = data.x[index];
        if (position === null) { continue; }
        var gap = Math.abs(position - x);
        if (gap < distance) { distance = gap; best = index; }
      }
      return best;
    }

    svg.addEventListener('pointermove', function (event) {
      var index = nearest(event);
      if (index < 0) { return; }
      var x = data.x[index];
      crosshair.setAttribute('x1', x);
      crosshair.setAttribute('x2', x);
      var rows = [];
      data.series.forEach(function (series, position) {
        var y = series.y[index];
        var dot = dots[position];
        var group = groups[position];
        if (y === null || y === undefined || (group && group.hasAttribute('hidden'))) {
          dot.setAttribute('r', '0');
          return;
        }
        dot.setAttribute('r', '4');
        dot.setAttribute('cx', x);
        dot.setAttribute('cy', y);
        rows.push({ name: series.name, value: series.labels[index], slot: series.slot });
      });
      layer.removeAttribute('hidden');
      showRows(data.xLabel + ' ' + data.xLabels[index], rows, event.clientX, event.clientY);
    });
    svg.addEventListener('pointerleave', function () {
      layer.setAttribute('hidden', 'hidden');
      hide();
    });
  });

  document.querySelectorAll('figure.chart').forEach(function (figure) {
    figure.querySelectorAll('button.legend-toggle').forEach(function (button) {
      button.addEventListener('click', function () {
        var index = button.getAttribute('data-series');
        var group = figure.querySelector('g.series[data-series="' + index + '"]');
        if (!group) { return; }
        var shown = button.getAttribute('aria-pressed') !== 'false';
        button.setAttribute('aria-pressed', shown ? 'false' : 'true');
        if (shown) { group.setAttribute('hidden', 'hidden'); }
        else { group.removeAttribute('hidden'); }
      });
    });
  });

  var zoom = null;
  var zoomed = null;
  var anchor = null;
  var factor = 1;

  function setFactor(next) {
    factor = Math.min(Math.max(next, 1), 8);
    if (zoomed) { zoomed.style.width = (factor * 100) + '%'; }
    zoom.querySelector('.zoom-level').textContent = Math.round(factor * 100) + '%';
  }

  function closeZoom() {
    if (zoomed && anchor) {
      anchor.parentNode.replaceChild(zoomed, anchor);
      zoomed.style.width = '';
      document.body.appendChild(tooltip);
    }
    zoomed = null;
    anchor = null;
    hide();
  }

  function buildZoom() {
    zoom = document.createElement('dialog');
    zoom.className = 'zoom';
    var head = document.createElement('div');
    head.className = 'zoom-head';
    var title = document.createElement('span');
    title.className = 'zoom-title';
    var controls = document.createElement('div');
    controls.className = 'zoom-controls';
    var level = document.createElement('span');
    level.className = 'zoom-level';
    var out = document.createElement('button');
    out.type = 'button';
    out.textContent = '−';
    out.setAttribute('aria-label', 'Zoom out');
    var into = document.createElement('button');
    into.type = 'button';
    into.textContent = '+';
    into.setAttribute('aria-label', 'Zoom in');
    var reset = document.createElement('button');
    reset.type = 'button';
    reset.textContent = 'Fit';
    var shut = document.createElement('button');
    shut.type = 'button';
    shut.textContent = 'Close';
    out.addEventListener('click', function () { setFactor(factor / 1.5); });
    into.addEventListener('click', function () { setFactor(factor * 1.5); });
    reset.addEventListener('click', function () { setFactor(1); });
    shut.addEventListener('click', function () { zoom.close(); });
    controls.appendChild(level);
    controls.appendChild(out);
    controls.appendChild(into);
    controls.appendChild(reset);
    controls.appendChild(shut);
    head.appendChild(title);
    head.appendChild(controls);
    var body = document.createElement('div');
    body.className = 'zoom-body';
    body.tabIndex = -1;
    zoom.appendChild(head);
    zoom.appendChild(body);
    zoom.addEventListener('close', closeZoom);
    zoom.addEventListener('click', function (event) {
      if (event.target === zoom) { zoom.close(); }
    });
    document.body.appendChild(zoom);
  }

  function openZoom(figure) {
    if (zoom === null) { buildZoom(); }
    if (zoomed) { zoom.close(); }
    var caption = figure.querySelector('.chart-title');
    zoom.querySelector('.zoom-title').textContent = caption ? caption.textContent : 'Chart';
    anchor = document.createElement('div');
    figure.parentNode.replaceChild(anchor, figure);
    zoomed = figure;
    var body = zoom.querySelector('.zoom-body');
    body.textContent = '';
    body.appendChild(figure);
    zoom.appendChild(tooltip);
    setFactor(1);
    if (typeof zoom.showModal === 'function') { zoom.showModal(); }
    body.scrollTop = 0;
    body.focus();
  }

  document.addEventListener('click', function (event) {
    var button = event.target.closest ? event.target.closest('button.chart-zoom') : null;
    if (button) { openZoom(button.closest('figure.chart')); }
  });

  var cells = null;
  function setMetric(metric) {
    if (cells === null) { cells = document.querySelectorAll('table.diff td'); }
    var attribute = 'data-' + metric;
    cells.forEach(function (cell) {
      var text = cell.getAttribute(attribute);
      if (text !== null) { cell.textContent = text; }
    });
    document.body.setAttribute('data-metric', metric);
  }
  document.querySelectorAll('input[name="metric"]').forEach(function (input) {
    input.addEventListener('change', function () {
      if (input.checked) { setMetric(input.value); }
    });
  });
  var metric = document.querySelector('input[name="metric"]:checked');
  if (metric && metric.value !== 'relative') { setMetric(metric.value); }

  var failOnly = document.getElementById('fail-only');
  var search = document.getElementById('search');
  function filter() {
    var only = failOnly && failOnly.checked;
    var query = search ? search.value.trim().toLowerCase() : '';
    document.querySelectorAll('table.diff tbody tr').forEach(function (row) {
      var name = (row.getAttribute('data-name') || '').toLowerCase();
      var failed = row.getAttribute('data-fail') === '1';
      row.hidden = (only && !failed) || (query !== '' && name.indexOf(query) === -1);
    });
    document.querySelectorAll('section.card').forEach(function (section) {
      var rows = section.querySelectorAll('table.diff tbody tr');
      if (!rows.length) { return; }
      var shown = section.querySelectorAll('table.diff tbody tr:not([hidden])').length;
      var counter = section.querySelector('.row-count');
      if (counter) { counter.textContent = shown + ' of ' + rows.length + ' shown'; }
    });
  }
  if (failOnly) { failOnly.addEventListener('change', filter); }
  if (search) { search.addEventListener('input', filter); }
  filter();

  var root = document.documentElement;
  try {
    var stored = localStorage.getItem('im-diff-theme');
    if (stored) { root.setAttribute('data-theme', stored); }
  } catch (error) { /* storage is unavailable, keep the system theme */ }
  var toggle = document.getElementById('theme-toggle');
  if (toggle) {
    toggle.addEventListener('click', function () {
      var current = root.getAttribute('data-theme');
      if (!current) {
        current = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
      }
      var next = current === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem('im-diff-theme', next); } catch (error) { /* ignore */ }
    });
  }
})();
"""


def _controls(comparison: Comparison) -> str:
    """Render the filter row and the colour key shared by every table.

    Parameters
    ----------
    comparison : Comparison
        The comparison being reported.

    Returns
    -------
    str
        The markup of the control row.
    """
    swatches = [
        ("var(--neg-3)", f"≤ -{_MAJOR_LEVEL:.0%}"),
        ("var(--neg-2)", f"-{_MAJOR_LEVEL:.0%} to -{_MINOR_LEVEL:.0%}"),
        ("var(--neg-1)", "smaller than benchmark"),
        ("var(--pos-1)", "larger than benchmark"),
        ("var(--pos-2)", f"{_MINOR_LEVEL:.0%} to {_MAJOR_LEVEL:.0%}"),
        ("var(--pos-3)", f"≥ {_MAJOR_LEVEL:.0%}"),
    ]
    keys = "".join(
        f'<span><i style="background: {colour}"></i>{_escape(label)}</span>'
        for colour, label in swatches
    )
    metrics = [
        ("relative", "Relative", "difference as a fraction of the benchmark value"),
        ("absolute", "Absolute", "difference in the unit of the intensity measure"),
        ("ulp", "ULP", "representable doubles between the two values"),
    ]
    options = "".join(
        f'<label title="{_escape(description)}">'
        f'<input type="radio" name="metric" value="{value}"'
        f"{' checked' if value == 'relative' else ''}> {_escape(label)}</label>"
        for value, label, description in metrics
    )
    return (
        '<section class="card controls">'
        f'<fieldset class="metric"><legend>Tables show</legend>{options}</fieldset>'
        '<label><input type="checkbox" id="fail-only"> Only values outside tolerance'
        f" (±{comparison.rtol:.3g} relative, ±{comparison.atol:.3g} absolute)</label>"
        '<label><input type="search" id="search" placeholder="Filter intensity measures"'
        ' aria-label="Filter intensity measures"></label>'
        f'<div class="swatches"><span class="swatch-note">Shading always follows the '
        f"relative difference:</span>{keys}</div>"
        "</section>"
    )


def render_diff_report(
    comparison: Comparison,
    title: str = "Intensity measure difference report",
    metadata: Mapping[str, str] | None = None,
) -> str:
    """Render a complete comparison as a standalone HTML document.

    Parameters
    ----------
    comparison : Comparison
        The comparison to render.
    title : str
        Title of the report.
    metadata : Mapping[str, str] | None
        Extra key/value pairs describing the run (waveform, core count, ...).

    Returns
    -------
    str
        The HTML document.
    """
    generated = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    statistics = summarise(comparison)
    details = {
        "generated": generated,
        "rows": " · ".join(str(label) for label in comparison.expected.index),
        "intensity measures": f"{comparison.expected.shape[1]:,}",
        "relative tolerance": f"{comparison.rtol:.3g}",
        "absolute tolerance": f"{comparison.atol:.3g}",
        **dict(metadata or {}),
    }

    scalar_columns = [
        column
        for family, columns in comparison.families.items()
        for column in columns
        if parse_column(column)[1] is None
    ]
    sections = [
        _summary_section(comparison, statistics, details),
        (
            '<section class="card"><h2>Where the tables differ</h2>'
            "<h3>By intensity measure family</h3>"
            f"{_family_summary_table(comparison)}"
            "<h3>Largest differences (values outside tolerance first)</h3>"
            f"{_worst_table(comparison)}</section>"
        ),
        _distribution_section(comparison),
        _controls(comparison),
    ]
    if scalar_columns:
        sections.append(_scalar_section(comparison, scalar_columns))
    for family, columns in comparison.families.items():
        if parse_column(columns[0])[1] is None:
            continue
        sections.append(_family_section(comparison, family, columns))

    subtitle = (
        f"{statistics.compared:,} values compared against the benchmark · "
        f"{statistics.failing:,} outside tolerance · generated {generated}"
    )
    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_escape(title)}</title><style>{_STYLESHEET}</style></head><body>"
        f'<header class="page-head"><div class="head-text"><h1>{_escape(title)}</h1>'
        f"<p>{_escape(subtitle)}</p></div>"
        '<button type="button" class="theme" id="theme-toggle">Toggle theme</button></header>'
        f"<main>{''.join(sections)}</main>"
        f"<script>{_SCRIPT}</script></body></html>"
    )


def write_diff_report(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    output_path: Path,
    title: str = "Intensity measure difference report",
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
    metadata: Mapping[str, str] | None = None,
) -> Path:
    """Write a standalone HTML report comparing two intensity measure tables.

    Parameters
    ----------
    expected : pd.DataFrame
        Benchmark values, indexed by component (or station) with one column per
        intensity measure.
    actual : pd.DataFrame
        Freshly calculated values in the same layout.
    output_path : Path
        File the report is written to; parent directories are created.
    title : str
        Title of the report.
    atol : float
        Absolute tolerance used to decide whether a value has changed.
    rtol : float
        Relative tolerance used to decide whether a value has changed.
    metadata : Mapping[str, str] | None
        Extra key/value pairs describing the run (waveform, core count, ...).

    Returns
    -------
    Path
        The path the report was written to.
    """
    comparison = Comparison.from_frames(expected, actual, atol=atol, rtol=rtol)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_diff_report(comparison, title=title, metadata=metadata)
    )
    return output_path
