"""Tests for the single-station `IM.im_calculation.calculate_ims` entry point."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import xarray as xr

from IM import im_calculation, ims

DT = 0.01
PERIODS = np.array([0.1, 0.5, 2.0])
FREQUENCIES = np.array([0.5, 1.0, 5.0])


@pytest.fixture(scope="module")
def station_waveform() -> npt.NDArray[np.float64]:
    """A single station's (1, nt, 3) waveform with distinct components."""
    rng = np.random.default_rng(1234)
    t = np.arange(0, 20, DT)
    envelope = np.exp(-(((t - 6) / 3) ** 2))
    waveform = np.stack(
        [
            envelope * rng.normal(scale=0.3, size=t.size),
            envelope * rng.normal(scale=0.2, size=t.size),
            envelope * rng.normal(scale=0.1, size=t.size),
        ],
        axis=-1,
    )
    return waveform[np.newaxis, :, :]


@pytest.fixture(scope="module")
def all_ims(station_waveform: npt.NDArray[np.float64]) -> pd.DataFrame:
    return im_calculation.calculate_ims(
        station_waveform, DT, periods=PERIODS, frequencies=FREQUENCIES
    )


def _component_waveform(
    station_waveform: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Reshape (1, nt, 3) into the (component, station, nt) layout of `ims`."""
    return np.ascontiguousarray(np.moveaxis(station_waveform, -1, 0))


def test_default_ims_produce_every_im_column(all_ims: pd.DataFrame) -> None:
    """With no `ims_list`, every IM appears, with one column per period/frequency."""
    expected_columns = (
        ["PGA", "PGV", "PGD", "CAV", "CAV5", "Ds575", "Ds595", "AI"]
        + [f"pSA_{p}" for p in PERIODS]
        + [f"FAS_{im_calculation.frequency_label(f)}" for f in FREQUENCIES]
    )
    assert list(all_ims.columns) == expected_columns
    # Rows are the components; the scalar IMs carry every one of them.
    for component in ("000", "090", "ver", "geom"):
        assert component in all_ims.index
    assert np.isfinite(
        all_ims.loc[["000", "090", "ver", "geom"], "PGA"].to_numpy()
    ).all()


@pytest.mark.parametrize(
    "im, compute",
    [
        (ims.IM.PGA, lambda w: ims.peak_ground_acceleration(w)),
        (ims.IM.PGV, lambda w: ims.peak_ground_velocity(w, DT)),
        (ims.IM.PGD, lambda w: ims.peak_ground_displacement(w, DT)),
        (ims.IM.CAV, lambda w: ims.cumulative_absolute_velocity(w, DT)),
        (
            ims.IM.CAV5,
            lambda w: ims.cumulative_absolute_velocity(w, DT, threshold=5),
        ),
        (ims.IM.Ds575, lambda w: ims.ds575(w, DT)),
        (ims.IM.Ds595, lambda w: ims.ds595(w, DT)),
        (ims.IM.AI, lambda w: ims.arias_intensity(w, DT)),
    ],
)
def test_scalar_im_matches_direct_computation(
    station_waveform: npt.NDArray[np.float64],
    im: ims.IM,
    compute: Callable[[npt.NDArray[np.float64]], xr.Dataset],
) -> None:
    """Each scalar IM is a single column equal to the underlying `ims` function."""
    result = im_calculation.calculate_ims(station_waveform, DT, [im])

    assert list(result.columns) == [im.value]
    expected = compute(_component_waveform(station_waveform))
    assert list(result.index) == list(expected.data_vars)
    for component in map(str, expected.data_vars):
        assert result.loc[component, im.value] == pytest.approx(
            float(expected[component].values[0]), nan_ok=True
        )


def test_psa_columns_match_direct_computation(
    station_waveform: npt.NDArray[np.float64],
) -> None:
    result = im_calculation.calculate_ims(
        station_waveform, DT, [ims.IM.pSA], periods=PERIODS
    )

    assert list(result.columns) == [f"pSA_{p}" for p in PERIODS]
    expected = ims.pseudo_spectral_acceleration(
        _component_waveform(station_waveform), PERIODS, DT
    )
    for component in map(str, expected.data_vars):
        np.testing.assert_allclose(
            result.loc[component].to_numpy(dtype=float),
            expected[component].values[0],
        )


def test_fas_uses_requested_bandwidth(
    station_waveform: npt.NDArray[np.float64],
) -> None:
    """The `bandwidth` argument is passed through to the Konno-Ohmachi smoothing."""
    bandwidth = 10.0
    result = im_calculation.calculate_ims(
        station_waveform,
        DT,
        [ims.IM.FAS],
        frequencies=FREQUENCIES,
        bandwidth=bandwidth,
    )
    default = im_calculation.calculate_ims(
        station_waveform, DT, [ims.IM.FAS], frequencies=FREQUENCIES
    )

    assert list(result.columns) == [
        f"FAS_{im_calculation.frequency_label(f)}" for f in FREQUENCIES
    ]
    expected = ims.fourier_amplitude_spectra(
        _component_waveform(station_waveform), DT, FREQUENCIES, bandwidth=bandwidth
    )
    for component in map(str, expected.data_vars):
        np.testing.assert_allclose(
            result.loc[component].to_numpy(dtype=float),
            expected[component].values[0],
        )
    # A different bandwidth smooths differently, so the spectra must differ.
    assert not np.allclose(
        result.loc["000"].to_numpy(dtype=float),
        default.loc["000"].to_numpy(dtype=float),
    )


def test_ims_list_order_and_selection_respected(
    station_waveform: npt.NDArray[np.float64], all_ims: pd.DataFrame
) -> None:
    """Only the requested IMs are computed, in the order requested."""
    result = im_calculation.calculate_ims(station_waveform, DT, [ims.IM.AI, ims.IM.PGA])
    assert list(result.columns) == ["AI", "PGA"]
    # The full run also has an FAS-only `eas` row; compare on the shared rows.
    assert "eas" not in result.index
    pd.testing.assert_frame_equal(result, all_ims.loc[result.index, ["AI", "PGA"]])


def test_unrecognised_im_raises(station_waveform: npt.NDArray[np.float64]) -> None:
    with pytest.raises(ValueError, match=r"IM NOT_AN_IM not recognized"):
        im_calculation.calculate_ims(
            station_waveform,
            DT,
            ["NOT_AN_IM"],  # ty: ignore[invalid-argument-type]  # deliberately not an IM
        )


@pytest.mark.parametrize("shape", [(1, 100, 2), (100,), (100, 4), (1, 1, 100, 3)])
def test_invalid_waveform_shape_message(shape: tuple[int, ...]) -> None:
    with pytest.raises(
        ValueError,
        match=r"^waveform must be a single station's waveform with shape "
        r"\(1, nt, 3\) or \(nt, 3\), but waveform\.shape=",
    ):
        im_calculation.calculate_ims(np.zeros(shape), DT, [ims.IM.PGA])


def test_dataset_to_frame_drops_non_dimension_coordinates() -> None:
    """Station metadata coordinates must not leak into the IM frame as rows/columns."""
    dataset = xr.Dataset(
        {
            "000": ("station", [1.0]),
            "090": ("station", [2.0]),
        },
        coords={"latitude": ("station", [-43.5]), "longitude": ("station", [172.6])},
    )

    frame = im_calculation._dataset_to_frame(dataset, ["PGA"])

    assert list(frame.columns) == ["000", "090"]
    assert list(frame.index) == ["PGA"]
    assert frame.loc["PGA", "000"] == 1.0
    assert frame.loc["PGA", "090"] == 2.0
