"""Test cases for intensity measure implementations."""

import functools
from collections.abc import Callable
from pathlib import Path

import dask.array as da
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import xarray as xr
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as nst
from numpy.testing import assert_array_almost_equal, assert_array_equal
from pytest import Metafunc, TempPathFactory

from IM import im_calculation, ims, snr_calculation, waveform_reading


@pytest.fixture(scope="session")
def ko_matrices(
    request: pytest.FixtureRequest, tmp_path_factory: TempPathFactory
) -> Path:
    from IM.scripts import gen_ko_matrix
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


def _to_dask(waveform: npt.NDArray[np.float64], station_chunk: int) -> xr.DataArray:
    """Wrap a bare waveform array as a dask-backed DataArray with real station
    names, chunked over `station` only (component/time as single chunks)."""
    n_stations = waveform.shape[1]
    return xr.DataArray(
        da.from_array(waveform, chunks=(waveform.shape[0], station_chunk, waveform.shape[2])),
        dims=("component", "station", "time"),
        coords={"station": [f"stat_{i}" for i in range(n_stations)]},
        attrs={"units": "g"},
    )


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
    result = ims.peak_ground_acceleration(waveforms)
    assert result.attrs["name"] == "PGA"
    assert np.isclose(result["000"].item(), expected_pga, atol=1e-3)


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
    ],
)
def test_pgv(
    comp_0: npt.NDArray[np.float64], t_max: float, expected_pgv: float
) -> None:
    waveforms = np.zeros((3, 1, len(comp_0)), dtype=np.float64)
    waveforms[ims.Component.COMP_0, 0, :] = comp_0
    dt = t_max / (len(comp_0) - 1)
    result = ims.peak_ground_velocity(waveforms, dt)
    assert result.attrs["name"] == "PGV"
    assert np.isclose(result["000"].item(), expected_pgv, atol=0.1)


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

    result = ims.cumulative_absolute_velocity(waveforms, dt, threshold=5)
    assert result.attrs["name"] == "CAV5"
    assert np.isclose(result["000"].item(), expected_cav5, atol=0.1)


def test_cav_name_depends_on_threshold(sample_waveforms: npt.NDArray[np.float64]) -> None:
    """threshold=0 is falsy, so it must name the result CAV, not CAV5."""
    assert ims.cumulative_absolute_velocity(sample_waveforms, 0.01).attrs["name"] == "CAV"
    assert (
        ims.cumulative_absolute_velocity(sample_waveforms, 0.01, threshold=0).attrs["name"]
        == "CAV"
    )
    assert (
        ims.cumulative_absolute_velocity(sample_waveforms, 0.01, threshold=5).attrs["name"]
        == "CAV5"
    )


@pytest.mark.slow
def test_fas_benchmark(ko_matrices: Path) -> None:
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
        waveform, dt, data.frequency.values, ko_matrices
    )

    for component in data.component.values:
        assert_array_almost_equal(
            data.sel(component=component).values,
            fas_result_ims[str(component)].values,
            decimal=5,
        )


def test_fas_multiple_stations_benchmark(ko_matrices: Path) -> None:
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
        duplicated_array, dt, data.frequency, ko_matrices
    )

    # Compare the results
    for component in data.component.values:
        expected = data.sel(component=component)[0, :]
        for station in range(2):
            assert_array_almost_equal(
                fas_result_ims[str(component)].isel(station=station).values,
                expected,
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

    # The benchmark predates the RotD orientation components and has no
    # reference angles to compare against; those are covered by
    # test_rotd_orientations_match_a_direct_angle_sweep instead.
    components = [component for component in result.index if component in data.index]
    for im in result.columns:
        assert result.loc[components, im].values == pytest.approx(
            data.loc[components, im].values, abs=5e-4, rel=0.01, nan_ok=True
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
            ]
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


@pytest.mark.slow
def test_all_ims_benchmark_edge_cases(resource_dir: Path, ko_matrices: Path) -> None:
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
        waveform, dt, ims_list=im_list, ko_directory=ko_matrices
    )

    # Align columns and indices for comparison, dropping the components the
    # benchmark does not carry (the RotD orientations, which postdate it).
    components = [component for component in result.index if component in data.index]
    result = result.loc[components]
    expected = data.loc[components, result.columns]

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
            expected,
            result,
            output_path=diff_path,
            title=f"Differences for {resource_dir.stem}",
        )

    # Perform standard assertions
    for im in result.columns:
        assert result[im].values == pytest.approx(
            expected[im].values, abs=5e-4, rel=0.01, nan_ok=True
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
    result = ims.significant_duration(sample_waveforms, dt, percent_low, percent_high)

    assert result.attrs["name"] == "duration"
    assert set(result.data_vars) == set(ims.GEOM_COMPONENTS)
    for component in ims.GEOM_COMPONENTS:
        assert result[component].shape == (sample_waveforms.shape[1],)
        assert np.all(result[component].values >= 0)
        assert np.all(result[component].values <= len(sample_time) * dt)


def test_ds5xx() -> None:
    comp_0 = np.ones((100,), dtype=np.float64)
    waveforms = np.zeros((3, 1, len(comp_0)), dtype=np.float64)
    waveforms[ims.Component.COMP_0, 0, :] = comp_0
    waveforms[ims.Component.COMP_90, 0, :] = comp_0 * 2
    waveforms[ims.Component.COMP_VER, 0, :] = comp_0 * 3
    dt = 1.0 / len(comp_0)

    ds575 = ims.ds575(waveforms, dt)
    ds595 = ims.ds595(waveforms, dt)
    assert ds575.attrs["name"] == "Ds575"
    assert ds595.attrs["name"] == "Ds595"
    assert ds575["000"].item() == pytest.approx(0.7)
    assert ds595["000"].item() == pytest.approx(0.9)


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
        result = func(sample_waveforms)
    else:
        result = func(sample_waveforms, dt)

    assert isinstance(result, xr.Dataset)
    assert set(result.data_vars) >= {"000", "090", "ver", "geom"}
    assert all((variable.values >= 0).all() for variable in result.data_vars.values())


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
    result = ims.fourier_amplitude_spectra(sample_waveforms, dt, freqs, ko_matrices)

    # Check Dataset structure
    assert isinstance(result, xr.Dataset)
    assert result.attrs["name"] == "FAS"
    assert list(result.data_vars) == list(ims.FAS_COMPONENTS)
    assert np.allclose(result.coords["frequency"], freqs)
    assert all((variable.values >= 0).all() for variable in result.data_vars.values())


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
    assert len(fas.data_vars) == 5
    for component in ims.FAS_COMPONENTS:
        assert fas[component].shape == (n_stations, len(expected_freqs)), (
            "Unexpected FAS shape."
        )


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

    with pytest.raises(TypeError):
        ims.peak_ground_acceleration(waveforms)  # ty: ignore[invalid-argument-type]


@pytest.mark.slow
def test_fourier_amplitude_spectra_shape(ko_matrices: Path) -> None:
    n_stations, n_timesteps, n_components = 2, 1024, 3
    dt = 0.01
    waveforms = np.random.rand(n_components, n_stations, n_timesteps).astype(np.float64)
    freqs = np.array([1.0, 10.0, 20.0], dtype=np.float64)

    fas = ims.fourier_amplitude_spectra(waveforms, dt, freqs, ko_matrices)
    assert len(fas.data_vars) == 5  # 5 components: 0, 90, ver, geom, eas
    for component in ims.FAS_COMPONENTS:
        assert fas[component].shape == (n_stations, len(freqs))


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
            ims.peak_ground_acceleration,
            functools.partial(ims.peak_ground_velocity, dt=0.01),
            functools.partial(
                ims.pseudo_spectral_acceleration,
                periods=np.array([1.0]),
                dt=0.01,
            ),
        ]
    ),
)
@settings(deadline=None)
@pytest.mark.slow
def test_rotational_invariance(
    waveform: npt.NDArray[np.float64],
    im: Callable[[ims.Waveform], xr.Dataset],
) -> None:
    old_waveform = np.copy(waveform)
    waveform_ims = im(old_waveform)
    assert np.allclose(old_waveform, waveform)
    waveform_ims_transposed = im(waveform[[1, 0, 2]])
    assert isinstance(waveform_ims, xr.Dataset)
    assert isinstance(waveform_ims_transposed, xr.Dataset)
    for component in ["rotd0", "rotd50", "rotd100"]:
        value = waveform_ims[component].values.squeeze()
        value_t = waveform_ims_transposed[component].values.squeeze()
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
    waveform_ims = ims.peak_ground_acceleration(waveform)

    assert_array_almost_equal(
        waveform_ims["000"].values,
        np.abs(waveform[ims.Component.COMP_0]).max(axis=1),
    )
    assert_array_almost_equal(
        waveform_ims["090"].values,
        np.abs(waveform[ims.Component.COMP_90]).max(axis=1),
    )
    assert_array_almost_equal(
        waveform_ims["ver"].values,
        np.abs(waveform[ims.Component.COMP_VER]).max(axis=1),
    )


def test_component_orientation_with_named_components(
    sample_waveforms: npt.NDArray[np.float64],
) -> None:
    """Component mapping is positional: index 0/1/2 -> 000/090/ver, regardless
    of how the input DataArray's `component` coordinate is labelled."""
    waveform = xr.DataArray(
        sample_waveforms,
        dims=("component", "station", "time"),
        coords={"component": ["x", "y", "z"]},
    )
    result = ims.peak_ground_acceleration(waveform)
    assert_array_almost_equal(
        result["000"].values, np.abs(sample_waveforms[0]).max(axis=-1)
    )
    assert_array_almost_equal(
        result["090"].values, np.abs(sample_waveforms[1]).max(axis=-1)
    )
    assert_array_almost_equal(
        result["ver"].values, np.abs(sample_waveforms[2]).max(axis=-1)
    )


# Lazy (dask-backed) input must produce a lazy Dataset whose computed values
# are bit-identical to the eager result -- station chunking never mixes rows,
# so nothing about laziness should change the numbers.
LAZY_CASES = [
    pytest.param(ims.peak_ground_acceleration, {}, id="pga"),
    pytest.param(ims.peak_ground_velocity, {"dt": 0.01}, id="pgv"),
    pytest.param(ims.peak_ground_displacement, {"dt": 0.01}, id="pgd"),
    pytest.param(ims.cumulative_absolute_velocity, {"dt": 0.01}, id="cav"),
    pytest.param(
        ims.cumulative_absolute_velocity, {"dt": 0.01, "threshold": 5}, id="cav5"
    ),
    pytest.param(ims.arias_intensity, {"dt": 0.01}, id="ai"),
    pytest.param(ims.ds575, {"dt": 0.01}, id="ds575"),
]


@pytest.mark.parametrize("func,kwargs", LAZY_CASES)
def test_lazy_matches_eager(
    sample_waveforms: npt.NDArray[np.float64],
    func: Callable[..., xr.Dataset],
    kwargs: dict,
) -> None:
    lazy_input = _to_dask(sample_waveforms, station_chunk=1)
    eager = func(sample_waveforms, **kwargs)
    lazy = func(lazy_input, **kwargs)

    assert all(v.chunks is not None for v in lazy.data_vars.values())
    assert "units" not in lazy.attrs  # keep_attrs=False: input attrs must not leak

    computed = lazy.compute()
    for component in eager.data_vars:
        assert_array_equal(eager[component].values, computed[component].values)


def test_lazy_matches_eager_psa(sample_waveforms: npt.NDArray[np.float64]) -> None:
    periods = np.array([0.1, 0.5, 1.0])
    lazy_input = _to_dask(sample_waveforms, station_chunk=1)
    eager = ims.pseudo_spectral_acceleration(sample_waveforms, periods, 0.01)
    lazy = ims.pseudo_spectral_acceleration(lazy_input, periods, 0.01)

    assert all(v.chunks is not None for v in lazy.data_vars.values())
    computed = lazy.compute()
    for component in eager.data_vars:
        assert_array_equal(eager[component].values, computed[component].values)


def test_psa_full_rotd180(sample_waveforms: npt.NDArray[np.float64]) -> None:
    """The full 180-angle curve must be internally consistent with the
    summary statistics computed from the same solve."""
    periods = np.array([0.1, 0.5, 1.0])
    dt = 0.01

    without = ims.pseudo_spectral_acceleration(sample_waveforms, periods, dt)
    with_curve = ims.pseudo_spectral_acceleration(
        sample_waveforms, periods, dt, full_rotd180=True
    )

    assert "rotd180" not in without.data_vars
    assert set(with_curve.data_vars) == set(without.data_vars) | {"rotd180"}
    assert with_curve["rotd180"].dims == ("station", "period", "angle")
    assert with_curve["rotd180"].shape == (
        sample_waveforms.shape[1],
        len(periods),
        180,
    )
    assert_array_equal(with_curve.angle.values, np.arange(180))

    # Angle 0 is exact (cos(0) == 1.0 exactly), so it must equal 000 exactly.
    assert_array_equal(with_curve["rotd180"].isel(angle=0).values, with_curve["000"].values)

    # The other summary components must be unaffected by asking for the curve.
    for component in without.data_vars:
        assert_array_equal(without[component].values, with_curve[component].values)

    # rotd0/50/100 must be exactly the min/median/max over the angle axis.
    curve = with_curve["rotd180"].values
    sorted_curve = np.sort(curve, axis=-1)
    assert_array_equal(sorted_curve[..., 0], with_curve["rotd0"].values)
    assert_array_equal(
        (sorted_curve[..., 89] + sorted_curve[..., 90]) / 2,
        with_curve["rotd50"].values,
    )
    assert_array_equal(sorted_curve[..., 179], with_curve["rotd100"].values)

    # And each orientation must be the angle of its own statistic in that same
    # curve: the argmin and argmax for rotd0/rotd100, and the lower of the two
    # central angles for rotd50, whose peak sits just below the reported
    # median.
    assert_array_equal(curve.argmin(axis=-1), with_curve["rotd0_orientation"].values)
    assert_array_equal(curve.argmax(axis=-1), with_curve["rotd100_orientation"].values)
    at_median = np.take_along_axis(
        curve,
        with_curve["rotd50_orientation"].values.astype(int)[..., np.newaxis],
        axis=-1,
    ).squeeze(-1)
    assert_array_equal(at_median, sorted_curve[..., 89])


def test_rotd_orientations_match_a_direct_angle_sweep(
    sample_waveforms: npt.NDArray[np.float64],
) -> None:
    """Each orientation must name the angle its statistic came from, against a
    plain numpy sweep of the two horizontal components."""
    result = ims.peak_ground_acceleration(sample_waveforms)
    comp_0 = sample_waveforms[ims.Component.COMP_0]
    comp_90 = sample_waveforms[ims.Component.COMP_90]

    angles = np.deg2rad(np.arange(180))
    # (n_stations, 180): the peak rotated amplitude at every integer angle.
    sweep = np.abs(
        np.cos(angles)[np.newaxis, :, np.newaxis] * comp_0[:, np.newaxis, :]
        + np.sin(angles)[np.newaxis, :, np.newaxis] * comp_90[:, np.newaxis, :]
    ).max(axis=-1)

    assert_array_equal(sweep.argmin(axis=-1), result["rotd0_orientation"].values)
    assert_array_equal(sweep.argmax(axis=-1), result["rotd100_orientation"].values)
    assert_array_equal(sweep.min(axis=-1), result["rotd0"].values)
    assert_array_equal(sweep.max(axis=-1), result["rotd100"].values)

    sorted_sweep = np.sort(sweep, axis=-1)
    at_median = np.take_along_axis(
        sweep,
        result["rotd50_orientation"].values.astype(int)[..., np.newaxis],
        axis=-1,
    ).squeeze(-1)
    assert_array_equal(at_median, sorted_sweep[..., 89])
    assert_array_equal(
        (sorted_sweep[..., 89] + sorted_sweep[..., 90]) / 2, result["rotd50"].values
    )


@pytest.mark.parametrize("polarisation", [0, 30, 45, 100, 179])
def test_rotd_orientation_of_a_polarised_record(polarisation: int) -> None:
    """A linearly polarised record fixes the orientations exactly: it peaks
    along its own direction and vanishes across it, which pins the angle
    convention (degrees, anticlockwise from the 000 component)."""
    time = np.arange(0, 1, 0.005)
    motion = np.sin(2 * np.pi * 5 * time) * np.exp(-2 * time)
    direction = np.deg2rad(polarisation)
    waveform = np.stack(
        [
            (motion * np.cos(direction))[np.newaxis],
            (motion * np.sin(direction))[np.newaxis],
            np.zeros((1, len(time))),
        ]
    )

    result = ims.peak_ground_acceleration(waveform)
    assert result["rotd100_orientation"].values == pytest.approx(polarisation)
    assert result["rotd0_orientation"].values == pytest.approx(
        (polarisation + 90) % 180
    )
    # Across the direction of motion there is nothing to see, and the sqrt(2)
    # bound on RotD100 / RotD50 is attained.
    assert result["rotd0"].values == pytest.approx(0, abs=1e-12)
    assert result["rotd100"].values / result["rotd50"].values == pytest.approx(
        np.sqrt(2), rel=1e-9
    )


def test_psa_full_rotd180_does_not_duplicate_the_solve(
    sample_waveforms: npt.NDArray[np.float64], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requesting the full curve must reuse the same per-period solve as the
    summary statistics, not run it a second time."""
    periods = np.array([0.1, 0.5, 1.0])
    dt = 0.01
    calls = []
    original = ims._core._psa_rotd180
    monkeypatch.setattr(
        ims._core,
        "_psa_rotd180",
        lambda *args, **kwargs: (calls.append(1), original(*args, **kwargs))[1],
    )

    ims.pseudo_spectral_acceleration(sample_waveforms, periods, dt, full_rotd180=False)
    n_without = len(calls)
    calls.clear()
    ims.pseudo_spectral_acceleration(sample_waveforms, periods, dt, full_rotd180=True)
    n_with = len(calls)

    assert n_without == len(periods)
    assert n_with == len(periods)


def test_lazy_matches_eager_psa_full_rotd180(
    sample_waveforms: npt.NDArray[np.float64],
) -> None:
    periods = np.array([0.1, 0.5, 1.0])
    lazy_input = _to_dask(sample_waveforms, station_chunk=1)
    eager = ims.pseudo_spectral_acceleration(
        sample_waveforms, periods, 0.01, full_rotd180=True
    )
    lazy = ims.pseudo_spectral_acceleration(
        lazy_input, periods, 0.01, full_rotd180=True
    )

    assert all(v.chunks is not None for v in lazy.data_vars.values())
    computed = lazy.compute()
    for component in eager.data_vars:
        assert_array_equal(eager[component].values, computed[component].values)


def test_lazy_matches_eager_fas(
    sample_waveforms: npt.NDArray[np.float64], ko_matrices: Path
) -> None:
    freqs = np.logspace(-1, 1, 16, dtype=np.float64)
    lazy_input = _to_dask(sample_waveforms, station_chunk=1)
    eager = ims.fourier_amplitude_spectra(sample_waveforms, 0.01, freqs, ko_matrices)
    lazy = ims.fourier_amplitude_spectra(lazy_input, 0.01, freqs, ko_matrices)

    assert all(v.chunks is not None for v in lazy.data_vars.values())
    computed = lazy.compute()
    # BLAS may re-block the Konno matmul differently per station chunk, so
    # allow a little slack rather than requiring bit-identical results.
    for component in eager.data_vars:
        assert_array_almost_equal(
            eager[component].values, computed[component].values, decimal=10
        )


def test_lazy_preserves_station_coord_and_extra_coords(
    sample_waveforms: npt.NDArray[np.float64],
) -> None:
    n_stations = sample_waveforms.shape[1]
    waveform = xr.DataArray(
        da.from_array(sample_waveforms, chunks=(3, 1, sample_waveforms.shape[2])),
        dims=("component", "station", "time"),
        coords={
            "station": [f"stat_{i}" for i in range(n_stations)],
            "latitude": ("station", np.arange(n_stations, dtype=np.float64)),
            "longitude": ("station", -np.arange(n_stations, dtype=np.float64)),
        },
    )
    result = ims.peak_ground_acceleration(waveform)
    assert_array_equal(result.station.values, waveform.station.values)
    assert_array_equal(result.latitude.values, waveform.latitude.values)
    assert_array_equal(result.longitude.values, waveform.longitude.values)


def test_rechunks_component_and_time_core_dims(
    sample_waveforms: npt.NDArray[np.float64],
) -> None:
    """A waveform chunked across `component`/`time` (as a real broadband file
    opened with `chunks={}` might be) must still work: `_as_waveform` forces
    those two dims back to a single chunk before `apply_ufunc` sees them."""
    waveform = xr.DataArray(
        da.from_array(sample_waveforms, chunks=(1, 1, 5)),
        dims=("component", "station", "time"),
    )
    result = ims.peak_ground_acceleration(waveform)
    computed = result.compute()
    expected = ims.peak_ground_acceleration(sample_waveforms)
    for component in expected.data_vars:
        assert_array_equal(expected[component].values, computed[component].values)
