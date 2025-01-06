"""Test cases for intensity measure implementations."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import functools
from numpy.testing import assert_array_almost_equal, assert_array_equal

from IM import ims

from hypothesis import assume, given, seed, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as nst


# Common test fixtures
@pytest.fixture
def sample_time():
    """Generate sample time array."""
    return np.arange(0, 1, 0.01, dtype=np.float32)


@pytest.fixture
def sample_waveforms():
    """Generate sample waveform data for testing."""
    t = np.arange(0, 1, 0.01, dtype=np.float32)
    freq = 5.0  # Hz
    stations = 2

    # Create synthetic acceleration time histories
    acc_0 = np.sin(2 * np.pi * freq * t)
    acc_90 = np.cos(2 * np.pi * freq * t)
    acc_ver = 0.5 * np.sin(2 * np.pi * freq * t)

    # Stack for multiple stations
    waveforms = np.zeros((stations, len(t), 3), dtype=np.float32)
    for i in range(stations):
        waveforms[i, :, ims.Component.COMP_0] = acc_0
        waveforms[i, :, ims.Component.COMP_90] = acc_90
        waveforms[i, :, ims.Component.COMP_VER] = acc_ver

    return waveforms


@pytest.fixture
def sample_periods():
    """Generate sample periods for PSA calculation."""
    return np.array([0.1, 0.2, 0.5, 1.0], dtype=np.float32)


# Test cases for Newmark PSA estimation
@pytest.mark.parametrize("xi", [0.01, 0.05, 0.1])
@pytest.mark.parametrize(
    "gamma,beta",
    [
        (0.5, 0.25),  # Standard Newmark
        (0.6, 0.3),  # Modified parameters
    ],
)
def test_newmark_estimate_psa(
    sample_waveforms: npt.NDArray[np.float32],
    sample_time: np.float32,
    xi: np.float32,
    gamma: np.float32,
    beta: np.float32,
):
    """Test Newmark PSA estimation with various parameters."""
    dt = sample_time[1] - sample_time[0]
    w = 2 * np.pi * np.array([1.0, 2.0], dtype=np.float32)

    result = ims.newmark_estimate_psa(
        sample_waveforms[:, :, ims.Component.COMP_0],
        sample_time,
        dt,
        w,
        xi=xi,
        gamma=gamma,
        beta=beta,
    )

    # Basic checks
    assert result.shape == (len(sample_waveforms), len(sample_time), len(w))
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))


# Test cases for rotd PSA values
def test_rotd_psa_values():
    """Test rotation of PSA values."""
    w = 2 * np.pi * np.array([1.0])
    comp_0 = np.atleast_3d(np.ones((2, 100), dtype=np.float32))
    comp_90 = np.atleast_3d(np.ones((2, 100), dtype=np.float32))

    result = ims.rotd_psa_values(comp_0, comp_90, w, 3)

    # Shape checks
    assert result.shape == (len(comp_0), len(w), 3)
    assert result[0, 0, 2] == pytest.approx((2 * np.pi) ** 2 * np.sqrt(2), abs=1e-3)


# Test cases for significant duration
@pytest.mark.parametrize(
    "percent_low,percent_high",
    [
        (5, 75),  # DS575
        (5, 95),  # DS595
        (20, 80),  # Custom range
    ],
)
def test_significant_duration(
    sample_waveforms: npt.NDArray[np.float32],
    sample_time: npt.NDArray[np.float32],
    percent_low: float,
    percent_high: float,
):
    """Test significant duration calculation."""
    dt = 0.01

    result = ims.significant_duration(
        sample_waveforms[:, :, 0], dt, percent_low, percent_high
    )

    # Basic checks
    assert result.shape == (len(sample_waveforms),)
    assert np.all(result >= 0)
    assert np.all(result <= len(sample_time) * dt)


# Test cases for peak ground motion parameters
@pytest.mark.parametrize(
    "func",
    [
        ims.peak_ground_acceleration,
        ims.peak_ground_velocity,
        ims.cumulative_absolute_velocity,
    ],
)
def test_peak_ground_parameters(
    sample_waveforms: npt.NDArray[np.float32],
    sample_time: npt.NDArray[np.float32],
    func: Callable,
):
    """Test peak ground motion parameter calculations."""
    dt = sample_time[1] - sample_time[0]

    if func == ims.peak_ground_acceleration:
        result = func(sample_waveforms)
    else:
        result = func(sample_waveforms, dt)

    # Check DataFrame structure
    assert isinstance(result, pd.DataFrame)
    assert all(
        col in result.columns
        for col in ["000", "090", "ver", "geom", "rotd50", "rotd100"]
    )
    # Check values
    assert np.all(result >= 0)  # All values should be non-negative


# Test cases for Arias Intensity
def test_arias_intensity(
    sample_waveforms: npt.NDArray[np.float32], sample_time: npt.NDArray[np.float32]
):
    """Test Arias Intensity calculation."""
    dt = sample_time[1] - sample_time[0]

    result = ims.arias_intensity(sample_waveforms, dt)

    # Check DataFrame structure
    assert isinstance(result, pd.DataFrame)
    assert all(
        col in result.columns
        for col in ["intensity_measure", "000", "090", "ver", "mean"]
    )
    # Check values
    assert np.all(result.select_dtypes(include=[np.number]) >= 0)
    # Check mean calculation
    assert_array_almost_equal(result["mean"], (result["000"] + result["090"]) / 2)


# Test cases for Fourier Amplitude Spectra
@pytest.mark.parametrize("n_freqs", [5, 10])
def test_fourier_amplitude_spectra(
    sample_waveforms: npt.NDArray[np.float32], sample_time: np.float32, n_freqs: int
):
    """Test Fourier Amplitude Spectra calculation."""
    dt = sample_time[1] - sample_time[0]
    freqs = np.logspace(-1, 1, n_freqs, dtype=np.float32)

    result = ims.fourier_amplitude_spectra(sample_waveforms, dt, freqs)

    # Check DataFrame structure
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ["freq", "000", "090", "ver", "mean"])
    # Check frequency values
    assert_array_equal(result["freq"].unique(), freqs)
    # Check values
    assert np.all(result.select_dtypes(include=[np.number]) >= 0)


# Error test cases
def test_invalid_memory_allocation():
    """Test handling of invalid memory allocation."""
    waveforms = np.zeros((2, 100, 3), dtype=np.float32)
    periods = np.array([0.1], dtype=np.float32)

    with pytest.raises(ValueError, match="PSA rotd memory allocation is too small"):
        ims.pseudo_spectral_acceleration(
            waveforms, periods, 0.01, psa_rotd_maximum_memory_allocation=1e-10
        )


@pytest.mark.parametrize(
    "invalid_shape",
    [
        (100,),  # 1D array
        (2, 100),  # Missing component dimension
    ],
)
def test_invalid_waveform_shapes(invalid_shape: tuple[int, ...]):
    """Test handling of invalid waveform shapes."""
    waveforms = np.zeros(invalid_shape, dtype=np.float32)

    with pytest.raises((ValueError, IndexError)):
        ims.peak_ground_acceleration(waveforms)


# Edge cases
def test_zero_waveform():
    """Test behavior with zero-amplitude waveforms."""
    waveforms = np.zeros((2, 100, 3), dtype=np.float32)
    dt = 0.01

    # Test various intensity measures with zero input
    pga_result = ims.peak_ground_acceleration(waveforms)
    ai_result = ims.arias_intensity(waveforms, dt)

    # All results should be zero
    assert np.all(pga_result.select_dtypes(include=[np.number]) == 0)
    assert np.all(ai_result.select_dtypes(include=[np.number]) == 0)


@pytest.mark.parametrize("duration", [100, 200, 1000])
def test_numerical_stability(duration: int):
    """Test numerical stability with different duration lengths."""
    dt = 0.01
    t = np.arange(0, duration * dt, dt, dtype=np.float32)
    waveforms = np.zeros((2, len(t), 3), dtype=np.float32)
    # Add sine wave
    waveforms[:, :, ims.Component.COMP_0] = np.sin(2 * np.pi / (duration * dt) * t)

    result = ims.peak_ground_acceleration(waveforms)

    # Maximum should be close to 1.0 regardless of duration
    assert_array_almost_equal(result["000"], 1.0, decimal=5)
    # The others should just be 0.0
    assert_array_almost_equal(result["090"], 0.0, decimal=5)


@given(
    func=st.sampled_from(
        [
            ims.peak_ground_acceleration,
            ims.peak_ground_velocity,
            ims.cumulative_absolute_velocity,
        ]
    ),
    waveform=nst.arrays(
        np.float32,
        shape=st.tuples(st.integers(2, 10), st.integers(2, 10), st.just(3)),
        elements=st.floats(-1, 1),
    ),
)
@settings(deadline=None)
def test_rotational_invariance(waveform: npt.NDArray[np.float32], func: Callable):
    if func != ims.peak_ground_acceleration:
        dt = 0.01
        func = functools.partial(func, dt=dt)
    waveform_ims = func(waveform)
    waveform_ims_transposed = func(waveform[:, :, [1, 0, 2]])

    assert_array_almost_equal(
        waveform_ims["rotd0"], waveform_ims_transposed["rotd0"], decimal=5
    )
    assert_array_almost_equal(
        waveform_ims["rotd50"], waveform_ims_transposed["rotd50"], decimal=5
    )
    assert_array_almost_equal(
        waveform_ims["rotd100"], waveform_ims_transposed["rotd100"], decimal=5
    )


@given(
    waveform=nst.arrays(
        np.float32,
        shape=st.tuples(st.integers(2, 10), st.integers(2, 10), st.just(3)),
        elements=st.floats(-1, 1),
    ),
)
@settings(deadline=None)
def test_component_orientation(waveform: npt.NDArray[np.float32]):
    waveform_ims = ims.peak_ground_acceleration(waveform)

    assert_array_almost_equal(
        waveform_ims["000"],
        np.abs(waveform[:, :, ims.Component.COMP_0]).max(axis=1),
    )
    assert_array_almost_equal(
        waveform_ims["090"],
        np.abs(waveform[:, :, ims.Component.COMP_90]).max(axis=1),
    )
    assert_array_almost_equal(
        waveform_ims["ver"],
        np.abs(waveform[:, :, ims.Component.COMP_VER]).max(axis=1),
    )
