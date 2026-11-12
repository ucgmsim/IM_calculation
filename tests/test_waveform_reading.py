"""Tests for ASCII waveform reading."""

from pathlib import Path

import numpy as np
import pytest

from IM import waveform_reading

VALUES_PER_LINE = 6


def write_component(path: Path, values: list[float], dt: float) -> Path:
    """Write a component in the ASCII format read by `read_ascii`."""
    lines = [
        "STAT       000",
        f"{len(values)}  {dt:.5e} 0 0  0.00000e+00  0.00000e+00  0.00000e+00  0.00000e+00",
    ]
    for i in range(0, len(values), VALUES_PER_LINE):
        lines.append(" ".join(f"{v:.5e}" for v in values[i : i + VALUES_PER_LINE]))
    path.write_text("\n".join(lines) + "\n")
    return path


def write_components(
    tmp_path: Path, components: list[list[float]], dt: float = 0.005
) -> list[Path]:
    """Write the 000, 090 and vertical components to `tmp_path`."""
    return [
        write_component(tmp_path / f"wave.{name}", values, dt)
        for name, values in zip(["000", "090", "ver"], components)
    ]


def test_strip_trailing_nans_removes_trailing_rows():
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [np.nan, np.nan, np.nan]])
    np.testing.assert_array_equal(waveform_reading.strip_trailing_nans(arr), arr[:2])


def test_strip_trailing_nans_keeps_middle_nans():
    arr = np.array([[1.0, 2.0, 3.0], [np.nan, 5.0, 6.0], [7.0, 8.0, 9.0]])
    np.testing.assert_array_equal(waveform_reading.strip_trailing_nans(arr), arr)


def test_strip_trailing_nans_keeps_rows_valid_in_any_column():
    # A row is only stripped when it is NaN in every column.
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, np.nan], [np.nan] * 3])
    np.testing.assert_array_equal(waveform_reading.strip_trailing_nans(arr), arr[:2])


@pytest.mark.parametrize("nt", [12, 14])
def test_read_ascii_dt_shape_and_values(tmp_path: Path, nt: int):
    # nt = 14 leaves a short last line, which the parser pads with NaN.
    components = [
        [float(i) + offset for i in range(nt)] for offset in (0.0, 100.0, 200.0)
    ]
    dt, waveform = waveform_reading.read_ascii(
        *write_components(tmp_path, components, dt=0.02)
    )

    assert dt == pytest.approx(0.02)
    assert waveform.shape == (1, nt, 3)
    assert waveform.dtype == np.float64
    np.testing.assert_array_equal(waveform[0], np.array(components).T)


def test_read_ascii_mid_record_nan_raises(tmp_path: Path):
    components = [[float(i) for i in range(12)] for _ in range(3)]
    components[1][5] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        waveform_reading.read_ascii(*write_components(tmp_path, components))


def test_read_ascii_unequal_component_lengths_raises(tmp_path: Path):
    # Pins current behaviour: the shorter component's trailing NaN is not
    # stripped because the other components are valid on that row.
    components = [
        [float(i) for i in range(14)],
        [float(i) for i in range(14)],
        [float(i) for i in range(13)],
    ]
    with pytest.raises(ValueError, match="NaN"):
        waveform_reading.read_ascii(*write_components(tmp_path, components))
