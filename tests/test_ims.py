"""Test cases for intensity measure implementations."""

import functools
import multiprocessing
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import xarray as xr
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as nst
from numpy.testing import assert_array_almost_equal
from rich import box
from rich.console import Console
from rich.table import Table

from IM import im_calculation, ims, snr_calculation, waveform_reading
from IM.scripts import gen_ko_matrix

KO_TEST_DIR = Path(__file__).parent / "KO_matrices"


@pytest.fixture(scope="session", autouse=True)
def generate_ko_matrices(request: pytest.FixtureRequest) -> None:
    KO_TEST_DIR.mkdir(exist_ok=True)
    gen_ko_matrix.main(KO_TEST_DIR, num_to_gen=12)

    def remove_ko_matrices() -> None:
        for file in KO_TEST_DIR.glob("*"):
            file.unlink()
        KO_TEST_DIR.rmdir()

    request.addfinalizer(remove_ko_matrices)


@pytest.fixture
def sample_time() -> npt.NDArray[np.float32]:
    return np.arange(0, 1, 0.01, dtype=np.float32)


@pytest.fixture
def sample_waveforms() -> npt.NDArray[np.float64]:
    """Generate sample waveform data in (n_components, n_stations, nt) shape."""
    t = np.arange(0, 1, 0.01, dtype=np.float64)
    freq = 5.0
    stations = 2

    acc_0 = np.sin(2 * np.pi * freq * t)
    acc_90 = 2 * np.cos(2 * np.pi * freq * t)
    acc_ver = 0.5 * np.sin(2 * np.pi * freq * t)

    waveforms = np.zeros((3, stations, len(t)), dtype=np.float64)
    for i in range(stations):
        waveforms[ims.Component.COMP_0, i, :] = acc_0
        waveforms[ims.Component.COMP_90, i, :] = acc_90
        waveforms[ims.Component.COMP_VER, i, :] = acc_ver

    return waveforms


@pytest.mark.parametrize("percent_low,percent_high", [(5, 75), (5, 95), (20, 80)])
def test_significant_duration(
    sample_waveforms: npt.NDArray[np.float64],
    sample_time: npt.NDArray[np.float32],
    percent_low: float,
    percent_high: float,
) -> None:
    dt = 0.01
    result = ims.significant_duration(
        sample_waveforms, dt, percent_low, percent_high, cores=1
    )

    assert result.shape == (sample_waveforms.shape[1], 4)  # 4 components
    assert np.all(result.values >= 0)
    assert np.all(result.values <= len(sample_time) * dt)


@pytest.mark.parametrize(
    "comp_0,expected_pga",
    [
        (np.ones((10,), dtype=np.float64), 1),
        (np.linspace(0, 1, num=10, dtype=np.float64) ** 2, 1),
        (2 * np.sin(np.linspace(0, 2 * np.pi, 50)) - 1, 3),
    ],
)
def test_pga(comp_0: npt.NDArray[np.float64], expected_pga: float) -> None:
    # Shape (n_comp, n_stat, nt)
    waveforms = np.zeros((3, 1, len(comp_0)), dtype=np.float64)
    waveforms[ims.Component.COMP_0, 0, :] = comp_0
    result = ims.peak_ground_acceleration(waveforms, cores=1)
    assert np.isclose(result["000"].iloc[0], expected_pga, atol=1e-3)


@pytest.mark.parametrize(
    "comp_0,t_max,expected_pgv",
    [
        (np.ones((100,), dtype=np.float64), 1, 981),
        (np.linspace(0, 1, num=100, dtype=np.float64) ** 2, 1, 981 / 3),
    ],
)
def test_pgv(
    comp_0: npt.NDArray[np.float64], t_max: float, expected_pgv: float
) -> None:
    waveforms = np.zeros((3, 1, len(comp_0)), dtype=np.float64)
    waveforms[ims.Component.COMP_0, 0, :] = comp_0
    dt = t_max / (len(comp_0) - 1)
    result = ims.peak_ground_velocity(waveforms, dt, cores=1)
    assert np.isclose(result["000"].iloc[0], expected_pgv, atol=0.1)


def test_psa() -> None:
    comp_0 = np.ones((100,), dtype=np.float64)
    waveforms = np.zeros((3, 2, len(comp_0)), dtype=np.float64)
    waveforms[ims.Component.COMP_0, 0, :] = comp_0
    waveforms[ims.Component.COMP_0, 1, :] = comp_0
    dt = np.float64(0.01)
    periods = np.array([0.1, 1.0], dtype=np.float64)

    psa_values = ims.pseudo_spectral_acceleration(waveforms, periods, dt, cores=1)

    # Check value for period 1.0 (Wolfram Alpha derived approx)
    assert psa_values.sel(
        station=0, period=1.0, component="000"
    ).item() == pytest.approx(1.8544671, abs=5e-3)


@pytest.mark.parametrize("cores", [1, 2])
def test_fas_benchmark(cores: int) -> None:
    data_array_ffp = Path(__file__).parent / "resources" / "fas_benchmark.nc"
    if not data_array_ffp.exists():
        pytest.skip("Benchmark file missing")
    data = xr.open_dataarray(data_array_ffp)

    data_dir = Path(__file__).parent.parent / "examples" / "resources"
    dt, waveform = waveform_reading.read_ascii(
        data_dir / "2024p950420_MWFS_HN_20.000",
        data_dir / "2024p950420_MWFS_HN_20.090",
        data_dir / "2024p950420_MWFS_HN_20.ver",
    )

    # Input: (n_stations, nt, n_components) as per fourier_amplitude_spectra logic
    fas_result_ims = ims.fourier_amplitude_spectra(
        waveform, dt, data.frequency.values, KO_TEST_DIR, cores=cores
    )

    assert_array_almost_equal(data.values, fas_result_ims.values, decimal=5)


def test_ds5xx() -> None:
    comp_0 = np.ones((100,), dtype=np.float64)
    waveforms = np.zeros((3, 1, len(comp_0)), dtype=np.float64)
    waveforms[ims.Component.COMP_0, 0, :] = comp_0
    waveforms[ims.Component.COMP_90, 0, :] = comp_0 * 2
    waveforms[ims.Component.COMP_VER, 0, :] = comp_0 * 3
    dt = 1.0 / len(comp_0)

    assert ims.ds575(waveforms, dt, cores=1)["000"].iloc[0] == pytest.approx(0.7)
    assert ims.ds595(waveforms, dt, cores=1)["000"].iloc[0] == pytest.approx(0.9)


@pytest.mark.parametrize(
    "func",
    [
        ims.peak_ground_acceleration,
        ims.peak_ground_velocity,
        ims.cumulative_absolute_velocity,
    ],
)
def test_peak_ground_parameters(
    sample_waveforms: npt.NDArray[np.float64],
    sample_time: npt.NDArray[np.float32],
    func: Callable,
) -> None:
    dt = float(sample_time[1] - sample_time[0])

    if func == ims.peak_ground_acceleration:
        result = func(sample_waveforms, cores=1)
    else:
        result = func(sample_waveforms, dt, cores=1)

    assert isinstance(result, pd.DataFrame)
    assert "geom" in result.columns
    assert np.all(result.select_dtypes(include=[np.number]) >= 0)


def test_fourier_amplitude_spectra_shape() -> None:
    n_stations, n_timesteps, n_components = 2, 1024, 3
    dt = 0.01
    waveforms = np.random.rand(n_stations, n_timesteps, n_components).astype(np.float64)
    freqs = np.array([1.0, 10.0, 20.0], dtype=np.float64)

    fas = ims.fourier_amplitude_spectra(waveforms, dt, freqs, KO_TEST_DIR, cores=1)
    assert fas.shape == (
        5,
        n_stations,
        len(freqs),
    )  # 5 components: 0, 90, ver, geom, eas


def test_invalid_waveform_shapes() -> None:
    waveforms = np.zeros((100,), dtype=np.float64)
    with pytest.raises(TypeError):
        ims.peak_ground_acceleration(waveforms, cores=1)


@given(
    waveform=nst.arrays(
        np.float64,
        shape=(3, 2, 10),  # (n_comp, n_stat, nt)
        elements=st.floats(0.1, 1),
    )
)
@settings(deadline=None)
def test_component_orientation(waveform: npt.NDArray[np.float64]) -> None:
    waveform_ims = ims.peak_ground_acceleration(waveform, cores=1)

    assert_array_almost_equal(
        waveform_ims["000"].values,
        np.abs(waveform[ims.Component.COMP_0, :, :]).max(axis=1),
    )
    assert_array_almost_equal(
        waveform_ims["ver"].values,
        np.abs(waveform[ims.Component.COMP_VER, :, :]).max(axis=1),
    )
