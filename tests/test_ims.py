"""Test cases for intensity measure implementations."""

import functools
import multiprocessing
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import xarray as xr
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as nst
from numpy.testing import assert_array_almost_equal
from pytest import Metafunc, TempPathFactory

from IM import im_calculation, ims, snr_calculation, waveform_reading
from IM.scripts import gen_ko_matrix


@pytest.fixture(scope="session", autouse=True)
def ko_matrices(
    request: pytest.FixtureRequest, tmp_path_factory: TempPathFactory
) -> Path:
    ko_matrix_directory = tmp_path_factory.mktemp("ko_matrices")
    gen_ko_matrix.main(ko_matrix_directory, num_to_gen=12)
    return ko_matrix_directory


@pytest.fixture
def sample_time() -> npt.NDArray[np.float64]:
    return np.arange(0, 1, 0.01, dtype=np.float64)


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


@pytest.fixture
def sample_periods() -> npt.NDArray[np.float64]:
    """Generate sample periods for PSA calculation."""
    return np.array([0.1, 0.2, 0.5, 1.0], dtype=np.float64)


# NOTE: The following unit tests PGA and PGV exist because there is no direct implementation of PGA/PGV in the rust code.


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
        (
            2 * np.sin(np.linspace(0, 2 * np.pi, num=100, dtype=np.float64)) - 1,
            2 * np.pi,
            981 * 2 * np.pi,
        ),
        (
            2 * np.sin(np.linspace(0, 2 * np.pi, num=100, dtype=np.float64)) - 1,
            2 * np.pi,
            981 * 2 * np.pi,
        ),
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


# CAV5 is partly a python function, so we test expected CAV5 results. CAV tests are in rust.
@pytest.mark.parametrize(
    "comp_0,t_max,expected_cav5",
    [
        (np.ones((100,), dtype=np.float64), 1, 9.81),
        (
            np.linspace(0, 1, num=100, dtype=np.float64) ** 2,
            1,
            9.81 / 3 * (1 - np.sqrt(5 / 981) ** 3),
        ),
    ],
)
def test_cav5(
    comp_0: npt.NDArray[np.float64], t_max: float, expected_cav5: float
) -> None:
    waveforms = np.zeros((3, 1, len(comp_0)), dtype=np.float64)
    waveforms[ims.Component.COMP_0] = comp_0
    dt = t_max / (len(comp_0) - 1)

    assert np.isclose(
        ims.cumulative_absolute_velocity(waveforms, dt, 1, threshold=5)["000"],
        expected_cav5,
        atol=0.1,
    )


@pytest.mark.parametrize("cores", [1, 2])
@pytest.mark.slow
def test_fas_benchmark(cores: int, ko_matrices: Path) -> None:
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
    waveform = np.ascontiguousarray(np.moveaxis(waveform, -1, 0))
    # Input: (n_stations, nt, n_components) as per fourier_amplitude_spectra logic
    fas_result_ims = ims.fourier_amplitude_spectra(
        waveform, dt, data.frequency.values, ko_matrices, cores=cores
    )

    assert_array_almost_equal(data.values, fas_result_ims.values, decimal=5)


@pytest.mark.parametrize("cores", [1, multiprocessing.cpu_count()])
def test_fas_multiple_stations_benchmark(cores: int, ko_matrices: Path) -> None:
    """Compare benchmark FAS calculation with multiple stations against current implementation."""
    # Load the data array
    data_array_ffp = Path(__file__).parent / "resources" / "fas_benchmark.nc"
    data = xr.open_dataarray(data_array_ffp)

    # Read the example waveform
    data_dir = Path(__file__).parent.parent / "examples" / "resources"
    comp_000_ffp = data_dir / "2024p950420_MWFS_HN_20.000"
    comp_090_ffp = data_dir / "2024p950420_MWFS_HN_20.090"
    comp_ver_ffp = data_dir / "2024p950420_MWFS_HN_20.ver"

    # Read the files to a waveform array that's readable by IM Calculation
    dt, waveform = waveform_reading.read_ascii(comp_000_ffp, comp_090_ffp, comp_ver_ffp)
    waveform = np.ascontiguousarray(np.moveaxis(waveform, -1, 0))
    # Duplicate the waveform array to simulate multiple stations (2 stations)
    duplicated_array = np.tile(waveform, (1, 2, 1))

    # Compute the Fourier Amplitude Spectra
    fas_result_ims = ims.fourier_amplitude_spectra(
        duplicated_array, dt, data.frequency, ko_matrices, cores=cores
    )

    # Compare the results
    for i in range(fas_result_ims.shape[1]):
        assert_array_almost_equal(
            fas_result_ims[:, i, :],
            data[:, 0, :],
            decimal=5,
        )


@pytest.mark.slow
def test_snr_benchmark(ko_matrices: Path) -> None:
    """Compare benchmark SNR calculation against current implementation."""
    # Load the DataFrame
    benchmark_ffp = Path(__file__).parent / "resources" / "snr_benchmark.csv"
    data = pd.read_csv(benchmark_ffp, index_col=0)

    # Read the example waveform
    data_dir = Path(__file__).parent.parent / "examples" / "resources"
    comp_000_ffp = data_dir / "2024p950420_MWFS_HN_20.000"
    comp_090_ffp = data_dir / "2024p950420_MWFS_HN_20.090"
    comp_ver_ffp = data_dir / "2024p950420_MWFS_HN_20.ver"

    # Read the files to a waveform array that's readable by IM Calculation
    dt, waveform = waveform_reading.read_ascii(comp_000_ffp, comp_090_ffp, comp_ver_ffp)
    waveform = np.ascontiguousarray(np.moveaxis(waveform, -1, 0))

    # Index of the start of the P-wave
    tp = 3170

    # Compute the SNR
    snr_result_ims, _, _, _, _ = snr_calculation.calculate_snr(
        waveform, dt, tp, ko_matrices
    )

    # Compare the results
    assert_array_almost_equal(
        data.values.astype(float), snr_result_ims.values.astype(float), decimal=5
    )


def test_all_ims_benchmark(ko_matrices: Path) -> None:
    """Compare benchmark IM calculation against current implementation."""
    # Load the DataFrame
    benchmark_ffp = Path(__file__).parent / "resources" / "im_benchmark.csv"
    data = pd.read_csv(benchmark_ffp, index_col=0)

    # Read the example waveform
    data_dir = Path(__file__).parent.parent / "examples" / "resources"
    comp_000_ffp = data_dir / "2024p950420_MWFS_HN_20.000"
    comp_090_ffp = data_dir / "2024p950420_MWFS_HN_20.090"
    comp_ver_ffp = data_dir / "2024p950420_MWFS_HN_20.ver"

    # Read the files to a waveform array that's readable by IM Calculation
    dt, waveform = waveform_reading.read_ascii(comp_000_ffp, comp_090_ffp, comp_ver_ffp)

    # Calculate the intensity measures
    result = im_calculation.calculate_ims(
        waveform,
        dt,
        ko_directory=ko_matrices,
    )

    for im in result.columns:
        assert result[im].values == pytest.approx(
            data.loc[result.index, im].values, abs=5e-4, rel=0.01, nan_ok=True
        ), (
            f"Results for {im} do not match!\n{result}"
        )  # 5e-6 implies rounding to five decimal places


# Assuming these are imported from your project context
# from your_module import waveform_reading, ims, im_calculation, BENCHMARK_CASES


def save_diff_html(
    df_old: pd.DataFrame,
    df_new: pd.DataFrame,
    output_path: Path,
    title: str = "DataFrame Difference",
) -> None:
    """
    Generates an HTML file highlighting differences between two dataframes.

    Cells are colored based on the relative difference:
    - >= 20%: Bold White on Red
    - > 0%: Black on Salmon
    - <= -20%: Bold White on Blue
    - < 0%: Black on Light Blue
    - No change / small change: Grey text
    """
    # Align dataframes to ensure dimensions match
    df_old, df_new = df_old.align(df_new, join="outer", axis=None)

    # Calculate differences
    diff_abs = df_new - df_old

    # Calculate relative difference safely
    with np.errstate(divide="ignore", invalid="ignore"):
        diff_rel = (df_new - df_old) / df_old

    def style_diff(data: pd.DataFrame) -> pd.DataFrame:
        """
        Styler function that receives the diff_abs DataFrame (data)
        but uses the diff_rel DataFrame from the outer scope to determine colors.
        """
        # Create a DataFrame of empty strings with same shape as data
        styles = pd.DataFrame("", index=data.index, columns=data.columns)

        # Ensure diff_rel aligns with the subset currently being styled
        # (Though with axis=None, 'data' is the full dataframe)
        rel_aligned = diff_rel.loc[data.index, data.columns]

        # --- Define Styles (Matching Rich output) ---
        style_high_pos = (
            "background-color: #d9534f; color: white; font-weight: bold;"  # Strong Red
        )
        style_low_pos = "background-color: #ffcccb; color: black;"  # Salmon
        style_high_neg = (
            "background-color: #0275d8; color: white; font-weight: bold;"  # Strong Blue
        )
        style_low_neg = "background-color: #add8e6; color: black;"  # Light Blue
        style_dim = "color: #999999;"  # Dim/Grey

        # --- Apply Logic ---

        # 1. Dim (Small changes or NaNs in relative diff)
        # Note: We use .fillna(False) to handle NaNs in the boolean mask creation
        is_small_change = (rel_aligned.abs() < 0.05) | (data.abs() < 1e-6)
        mask_dim = is_small_change | rel_aligned.isna()
        styles[mask_dim] = style_dim

        # 2. Strong Positive (>= 20%)
        mask_high_pos = (rel_aligned >= 0.20) & ~mask_dim
        styles[mask_high_pos] = style_high_pos

        # 3. Low Positive (> 0 and < 20%)
        mask_low_pos = (rel_aligned > 0) & (rel_aligned < 0.20) & ~mask_dim
        styles[mask_low_pos] = style_low_pos

        # 4. Strong Negative (<= -20%)
        mask_high_neg = (rel_aligned <= -0.20) & ~mask_dim
        styles[mask_high_neg] = style_high_neg

        # 5. Low Negative (< 0 and > -20%)
        mask_low_neg = (rel_aligned < 0) & (rel_aligned > -0.20) & ~mask_dim
        styles[mask_low_neg] = style_low_neg

        return styles

    # Create the Styler object
    # We display diff_abs, but color it based on relative diff logic
    styler = (
        diff_abs.style.apply(style_diff, axis=None)
        .format("{:+.3g}", na_rep="-")
        .set_caption(title)
        .set_table_styles(
            [
                {
                    "selector": "caption",
                    "props": [("font-size", "1.5em"), ("font-weight", "bold")],
                },
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#f2f2f2"),
                        ("text-align", "center"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [("padding", "5px"), ("border", "1px solid #ddd")],
                },
            ]  # type: ignore[invalid-argument-type]
        )
    )

    # Save to file
    with open(output_path, "w") as f:
        f.write(styler.to_html())


def pytest_generate_tests(metafunc: Metafunc) -> None:
    if "resource_dir" in metafunc.fixturenames:
        benchmark_cases = [
            d for d in (Path(__file__).parent / "resources").iterdir() if d.is_dir()
        ]
        metafunc.parametrize("resource_dir", benchmark_cases, ids=lambda p: p.stem)


@pytest.mark.parametrize("cores", [1, multiprocessing.cpu_count()])
@pytest.mark.slow
def test_all_ims_benchmark_edge_cases(
    resource_dir: Path, cores: int, ko_matrices: Path
) -> None:
    """Compare benchmark IM calculation against current implementation for each directory in resources for edge cases."""
    # Load the benchmark DataFrame
    benchmark_ffp = resource_dir / "im_benchmark.csv"
    data = pd.read_csv(benchmark_ffp, index_col=0)

    # Read the edge case waveform files
    comp_000_ffp = resource_dir / f"{resource_dir.stem}.000"
    comp_090_ffp = resource_dir / f"{resource_dir.stem}.090"
    comp_ver_ffp = resource_dir / f"{resource_dir.stem}.ver"

    # Read the files to a waveform array that's readable by IM Calculation
    dt, waveform = waveform_reading.read_ascii(comp_000_ffp, comp_090_ffp, comp_ver_ffp)
    nt = waveform.shape[1]

    im_list = [
        ims.IM.PGA,
        ims.IM.PGV,
        ims.IM.CAV,
        ims.IM.CAV5,
        ims.IM.Ds575,
        ims.IM.Ds595,
        ims.IM.AI,
        ims.IM.pSA,
    ]

    # If the record is too long the test will fail because of missing KO matrices
    have_ko_matrix = np.ceil(np.log2(nt)) < 15
    if have_ko_matrix:
        im_list.append(ims.IM.FAS)

    # Calculate the intensity measures
    result = im_calculation.calculate_ims(
        waveform, dt, ims_list=im_list, ko_directory=ko_matrices, cores=cores
    )

    # Align columns and indices for comparison
    expected = data.loc[result.index, result.columns]

    # Check for failure
    if not np.allclose(
        result.values, expected.values, atol=5e-4, rtol=0.01, equal_nan=True
    ):
        # Define output filename
        diff_filename = f"diff_fail_{resource_dir.stem}.html"
        diff_path = Path.cwd() / diff_filename  # Or use a specific artifacts dir

        print(f"\n[!] Benchmark mismatch. Saving HTML diff report to: {diff_path}")

        # Select only pSA columns as per original logic, or remove filter to show all
        save_diff_html(
            expected,  # type: ignore[invalid-argument]
            result,
            output_path=diff_path,
            title=f"Differences for {resource_dir.stem}",
        )

    # Perform standard assertions
    for im in result.columns:
        assert result[im].values == pytest.approx(
            data.loc[result.index, im].values, abs=5e-4, rel=0.01, nan_ok=True
        ), f"Results for {im} do not match!\n{result}"


# Significant duration calculations are a combination of two
# independently tested rust functions, this integration test checks
# they are called correctly through the python interface


@pytest.mark.parametrize("percent_low,percent_high", [(5, 75), (5, 95), (20, 80)])
def test_significant_duration(
    sample_waveforms: npt.NDArray[np.float64],
    sample_time: npt.NDArray[np.float64],
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


def test_ds5xx() -> None:
    comp_0 = np.ones((100,), dtype=np.float64)
    waveforms = np.zeros((3, 1, len(comp_0)), dtype=np.float64)
    waveforms[ims.Component.COMP_0, 0, :] = comp_0
    waveforms[ims.Component.COMP_90, 0, :] = comp_0 * 2
    waveforms[ims.Component.COMP_VER, 0, :] = comp_0 * 3
    dt = 1.0 / len(comp_0)

    assert ims.ds575(waveforms, dt, cores=1)["000"].iloc[0] == pytest.approx(0.7)
    assert ims.ds595(waveforms, dt, cores=1)["000"].iloc[0] == pytest.approx(0.9)


# Contract guarantee on output shapes
@pytest.mark.parametrize(
    "func",
    [
        ims.peak_ground_acceleration,
        ims.peak_ground_velocity,
        ims.arias_intensity,
        ims.cumulative_absolute_velocity,
    ],
)
def test_peak_ground_parameters(
    sample_waveforms: npt.NDArray[np.float64],
    sample_time: npt.NDArray[np.float64],
    func: Callable,
) -> None:
    dt = float(sample_time[1] - sample_time[0])

    if func == ims.peak_ground_acceleration:
        result = func(sample_waveforms, cores=1)
    else:
        result = func(sample_waveforms, dt, cores=1)

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"000", "090", "ver", "geom"}
    assert np.all(result.select_dtypes(include=[np.number]) >= 0)


# Test cases for Fourier Amplitude Spectra
@pytest.mark.parametrize("n_freqs", [1024, 2048])
def test_fourier_amplitude_spectra(
    sample_waveforms: npt.NDArray[np.float64],
    sample_time: npt.NDArray[np.float64],
    ko_matrices: Path,
    n_freqs: int,
) -> None:
    """Test Fourier Amplitude Spectra calculation."""
    dt = sample_time[1] - sample_time[0]
    freqs = np.logspace(-1, 1, n_freqs, dtype=np.float64)
    # Force the multiprocessing code path if necessary.
    result_mp = ims.fourier_amplitude_spectra(
        sample_waveforms,
        dt,
        freqs,
        ko_matrices,
        cores=max(2, multiprocessing.cpu_count()),
    )
    # Force the single core path.
    result_sc = ims.fourier_amplitude_spectra(
        sample_waveforms, dt, freqs, ko_matrices, cores=1
    )

    # Check DataFrame structure
    assert isinstance(result_mp, xr.DataArray)
    assert list(result_mp.coords["component"]) == ["000", "090", "ver", "geom", "eas"]
    assert np.allclose(result_mp.coords["frequency"], freqs)
    assert np.all(result_mp.as_numpy() >= 0)
    # Check that multi-core result and single-core result produce the same output.
    assert np.allclose(result_mp.as_numpy(), result_sc.as_numpy())


def test_nyquist_frequency(ko_matrices: Path) -> None:
    # Define test parameters
    n_stations = 2
    n_timesteps = 1024
    n_components = 3
    dt = 0.01  # Timestep resolution (s)
    nyquist_frequency = 1 / (2 * dt)

    # Generate test waveforms (random data for simplicity)
    waveforms = np.random.rand(n_components, n_stations, n_timesteps).astype(np.float64)

    # Define frequencies, including some above the Nyquist frequency
    freqs = np.array(
        [1.0, 10.0, 20.0, 60.0], dtype=np.float64
    )  # 60 Hz > Nyquist (50 Hz)
    with pytest.warns(RuntimeWarning):
        fas = ims.fourier_amplitude_spectra(waveforms, dt, freqs, ko_matrices)

    # Verify that frequencies above Nyquist are filtered out
    expected_freqs = freqs[freqs <= nyquist_frequency]
    np.testing.assert_array_equal(fas.coords["frequency"].values, expected_freqs)

    # Verify the shape of the output
    assert fas.shape == (5, n_stations, len(expected_freqs)), "Unexpected FAS shape."


@pytest.mark.parametrize(
    "invalid_shape",
    [
        (100,),  # 1D array
        (2, 100),  # Missing component dimension
    ],
)
def test_invalid_waveform_shapes(invalid_shape: tuple[int, ...]) -> None:
    """Test handling of invalid waveform shapes."""
    waveforms = np.zeros(invalid_shape, dtype=np.float64)

    with pytest.raises((TypeError)):
        ims.peak_ground_acceleration(waveforms, cores=1)  # type: ignore[invalid-argument-type]


@pytest.mark.slow
def test_fourier_amplitude_spectra_shape(ko_matrices: Path) -> None:
    n_stations, n_timesteps, n_components = 2, 1024, 3
    dt = 0.01
    waveforms = np.random.rand(n_components, n_stations, n_timesteps).astype(np.float64)
    freqs = np.array([1.0, 10.0, 20.0], dtype=np.float64)

    fas = ims.fourier_amplitude_spectra(waveforms, dt, freqs, ko_matrices, cores=1)
    assert fas.shape == (
        5,
        n_stations,
        len(freqs),
    )  # 5 components: 0, 90, ver, geom, eas


# Asserts that the RotDx values of PGA, PGV and pSA are invariant of the order of 000 and 090.
@given(
    waveform=nst.arrays(
        np.float64,
        shape=st.tuples(st.just(3), st.integers(2, 10), st.integers(10, 100)),
        elements=st.floats(0.01, 1, width=64).flatmap(
            lambda x: st.sampled_from([-1, 1]).flatmap(lambda sign: st.just(sign * x))
        ),
    ),
    im=st.sampled_from(
        [
            functools.partial(ims.peak_ground_acceleration, cores=1),
            functools.partial(ims.peak_ground_velocity, dt=0.01, cores=1),
            functools.partial(
                ims.pseudo_spectral_acceleration,
                periods=np.array([1.0]),
                dt=0.01,
                cores=1,
            ),
        ]
    ),
)
@settings(deadline=None)
@pytest.mark.slow
def test_rotational_invariance(
    waveform: npt.NDArray[np.float64],
    im: Callable[[ims.ChunkedWaveformArray], pd.DataFrame | xr.DataArray],
) -> None:
    old_waveform = np.copy(waveform)
    waveform_ims = im(old_waveform)
    assert np.allclose(old_waveform, waveform)
    waveform_ims_transposed = im(waveform[[1, 0, 2]])
    if isinstance(waveform_ims_transposed, pd.DataFrame) and isinstance(
        waveform_ims, pd.DataFrame
    ):
        for component in ["rotd0", "rotd50", "rotd100"]:
            value = waveform_ims[component].values
            value_t = waveform_ims_transposed[component].values
            assert value == pytest.approx(value_t)
    else:
        assert isinstance(waveform_ims, xr.DataArray)
        assert isinstance(waveform_ims_transposed, xr.DataArray)
        for component in ["rotd0", "rotd50", "rotd100"]:
            value = waveform_ims.sel(component=component).values.squeeze()
            value_t = waveform_ims_transposed.sel(component=component).values.squeeze()
            assert value == pytest.approx(value_t)


# Asserts that 090, 000, and ver components are computed for the corresponding COMP_* enum values.
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
        waveform_ims["000"].values,  # type: ignore[invalid-argument-type]
        np.abs(waveform[ims.Component.COMP_0]).max(axis=1),
    )
    assert_array_almost_equal(
        waveform_ims["090"].values,  # type: ignore[invalid-argument-type]
        np.abs(waveform[ims.Component.COMP_90]).max(axis=1),
    )
    assert_array_almost_equal(
        waveform_ims["ver"].values,  # type: ignore[invalid-argument-type]
        np.abs(waveform[ims.Component.COMP_VER]).max(axis=1),
    )
