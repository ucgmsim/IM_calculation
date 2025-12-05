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
    """
    Generate the KO matrices for testing, also test that the KO matrix gen script works.
    """
    # Make the KO matrices directory
    KO_TEST_DIR.mkdir(exist_ok=True)
    gen_ko_matrix.main(KO_TEST_DIR, num_to_gen=12)

    # Add finalizer to remove the KO matrices directory
    def remove_ko_matrices() -> None:
        for file in KO_TEST_DIR.glob("*"):
            file.unlink()
        KO_TEST_DIR.rmdir()

    request.addfinalizer(remove_ko_matrices)


# Common test fixtures
@pytest.fixture
def sample_time() -> npt.NDArray[np.float32]:
    """Generate sample time array."""
    return np.arange(0, 1, 0.01, dtype=np.float32)


@pytest.fixture
def sample_waveforms() -> npt.NDArray[np.float32]:
    """Generate sample waveform data for testing."""
    t = np.arange(0, 1, 0.01, dtype=np.float32)
    freq = 5.0  # Hz
    stations = 2

    # Create synthetic acceleration time histories
    acc_0 = np.sin(2 * np.pi * freq * t)
    acc_90 = 2 * np.cos(2 * np.pi * freq * t)
    acc_ver = 0.5 * np.sin(2 * np.pi * freq * t)

    # Stack for multiple stations
    waveforms = np.zeros((stations, len(t), 3), dtype=np.float32)
    for i in range(stations):
        waveforms[i, :, ims.Component.COMP_0] = acc_0
        waveforms[i, :, ims.Component.COMP_90] = acc_90
        waveforms[i, :, ims.Component.COMP_VER] = acc_ver

    return waveforms


@pytest.fixture
def sample_periods() -> npt.NDArray[np.float32]:
    """Generate sample periods for PSA calculation."""
    return np.array([0.1, 0.2, 0.5, 1.0], dtype=np.float32)


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
) -> None:
    """Test significant duration calculation."""
    dt = 0.01

    result = ims.significant_duration(
        sample_waveforms[:, :, 0], dt, percent_low, percent_high
    )

    # Basic checks
    assert result.shape == (len(sample_waveforms),)
    assert np.all(result >= 0)
    assert np.all(result <= len(sample_time) * dt)


@pytest.mark.parametrize(
    "comp_0,expected_pga,use_numexpr",
    [
        (np.ones((3,), dtype=np.float32), 1, True),
        (np.linspace(0, 1, num=10, dtype=np.float32) ** 2, 1, True),
        (2 * np.sin(np.linspace(0, 2 * np.pi)) - 1, 3, True),
        (2 * np.sin(np.linspace(0, 2 * np.pi)) - 1, 3, False),
    ],
)
def test_pga(
    comp_0: npt.NDArray[np.float32], expected_pga: float, use_numexpr: bool
) -> None:
    waveforms = np.zeros((1, len(comp_0), 3), dtype=np.float32)
    waveforms[:, :, ims.Component.COMP_0] = comp_0
    assert np.isclose(
        ims.peak_ground_acceleration(waveforms, use_numexpr=use_numexpr)["000"],
        expected_pga,
        atol=1e-3,
    )


@pytest.mark.parametrize(
    "comp_0,t_max,expected_pga,use_numexpr",
    [
        (np.ones((100,), dtype=np.float32), 1, 981, True),
        (np.linspace(0, 1, num=100, dtype=np.float32) ** 2, 1, 981 / 3, True),
        (
            2 * np.sin(np.linspace(0, 2 * np.pi, num=100, dtype=np.float32)) - 1,
            2 * np.pi,
            981 * 2 * np.pi,
            True,
        ),
        (
            2 * np.sin(np.linspace(0, 2 * np.pi, num=100, dtype=np.float32)) - 1,
            2 * np.pi,
            981 * 2 * np.pi,
            False,
        ),
    ],
)
def test_pgv(
    comp_0: npt.NDArray[np.float32],
    t_max: float,
    expected_pga: float,
    use_numexpr: bool,
) -> None:
    waveforms = np.zeros((1, len(comp_0), 3), dtype=np.float32)
    waveforms[:, :, ims.Component.COMP_0] = comp_0
    # NOTE: This dt calculation is correct, if dt = 1 / len(comp_0) then dt
    # ends up *too small* and these tests will fail.
    dt = t_max / (len(comp_0) - 1)
    assert np.isclose(
        ims.peak_ground_velocity(waveforms, dt, use_numexpr=use_numexpr)["000"],
        expected_pga,
        atol=0.1,
    )


@pytest.mark.parametrize(
    "comp_0,t_max,expected_cav,expected_cav5",
    [
        (np.ones((100,), dtype=np.float32), 1, 9.81, 9.81),
        (
            np.linspace(0, 1, num=100, dtype=np.float32) ** 2,
            1,
            9.81 / 3,
            9.81 / 3 * (1 - np.sqrt(5 / 981) ** 3),
        ),
        (
            2 * np.sin(np.linspace(0, 2 * np.pi, num=1000, dtype=np.float32)) - 1,
            2 * np.pi,
            9.81 * 2 / 3 * (6 * np.sqrt(3) + np.pi),
            None,
        ),
    ],
)
def test_cav(
    comp_0: npt.NDArray[np.float32],
    t_max: float,
    expected_cav: float,
    expected_cav5: Optional[float],
) -> None:
    waveforms = np.zeros((1, len(comp_0), 3), dtype=np.float32)
    waveforms[:, :, ims.Component.COMP_0] = comp_0
    dt = t_max / (len(comp_0) - 1)
    assert np.isclose(
        ims.cumulative_absolute_velocity(waveforms, dt)["000"], expected_cav, atol=0.1
    )
    if expected_cav5:
        assert np.isclose(
            ims.cumulative_absolute_velocity(waveforms, dt, threshold=5)["000"],
            expected_cav5,
            atol=0.1,
        )


@pytest.mark.parametrize(
    "comp_0,t_max,expected_ai",
    [
        (np.ones((100,), dtype=np.float32), 1, np.pi * 9.81 / 2),
        (np.linspace(0, 1, num=100, dtype=np.float32) ** 2, 1, np.pi * 9.81 / (2 * 5)),
    ],
)
def test_ai_values(
    comp_0: npt.NDArray[np.float32], t_max: float, expected_ai: float
) -> None:
    waveforms = np.zeros((1, len(comp_0), 3), dtype=np.float32)
    waveforms[:, :, ims.Component.COMP_0] = comp_0
    dt = t_max / (len(comp_0) - 1)
    assert np.isclose(ims.arias_intensity(waveforms, dt)["000"], expected_ai, atol=0.1)


def test_psa() -> None:
    comp_0 = np.ones((100,), dtype=np.float32)
    waveforms = np.zeros((3, 2, len(comp_0)), dtype=np.float32)
    waveforms[ims.Component.COMP_0, 0] = comp_0
    waveforms[ims.Component.COMP_0, 1] = comp_0
    dt = np.float64(0.01)
    w = np.array([1, 2], dtype=np.float64)
    psa_values = ims.pseudo_spectral_acceleration(
        waveforms,
        w,
        dt,
    )

    # assert psa is close to the expected psa derived by solving the ODE in
    # Wolfram Alpha and finding the abs max.
    assert psa_values.sel(
        station=0, period=1.0, component="000"
    ).item() == pytest.approx(1.8544671, abs=5e-3)


@pytest.mark.parametrize("cores", [1, multiprocessing.cpu_count()])
def test_fas_benchmark(cores: int) -> None:
    """Compare benchmark FAS calculation against current implementation."""
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

    # Compute the Fourier Amplitude Spectra
    fas_result_ims = ims.fourier_amplitude_spectra(
        waveform, dt, data.frequency, KO_TEST_DIR, cores=cores
    )

    # Compare the results
    assert_array_almost_equal(data, fas_result_ims, decimal=5)


@pytest.mark.parametrize("cores", [1, multiprocessing.cpu_count()])
def test_fas_multiple_stations_benchmark(cores: int) -> None:
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

    # Duplicate the waveform array to simulate multiple stations (2 stations)
    duplicated_array = np.tile(waveform, (2, 1, 1))

    # Compute the Fourier Amplitude Spectra
    fas_result_ims = ims.fourier_amplitude_spectra(
        duplicated_array, dt, data.frequency, KO_TEST_DIR, cores=cores
    )

    # Compare the results
    for i in range(fas_result_ims.shape[1]):
        assert_array_almost_equal(
            fas_result_ims[:, i, :],
            data[:, 0, :],
            decimal=5,
        )


def test_snr_benchmark() -> None:
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

    # Index of the start of the P-wave
    tp = 3170

    # Compute the SNR
    snr_result_ims, _, _, _, _ = snr_calculation.calculate_snr(
        waveform, dt, tp, KO_TEST_DIR
    )

    # Compare the results
    assert_array_almost_equal(
        data.values.astype(float), snr_result_ims.values.astype(float), decimal=5
    )


@pytest.mark.parametrize("use_numexpr", [True, False])
def test_all_ims_benchmark(use_numexpr: bool) -> None:
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
        waveform, dt, ko_directory=KO_TEST_DIR, use_numexpr=use_numexpr
    )

    for im in result.columns:
        assert result[im].values == pytest.approx(
            data[im].values, abs=5e-4, rel=0.01, nan_ok=True
        ), (
            f"Results for {im} do not match!\n{result}"
        )  # 5e-6 implies rounding to five decimal places


BENCHMARK_CASES = [
    d for d in (Path(__file__).parent / "resources").iterdir() if d.is_dir()
]


def print_diff_table(
    df_old: pd.DataFrame,
    df_new: pd.DataFrame,
    title: str = "DataFrame Difference",
    chunk_size: int | None = None,
) -> None:
    """Prints a Rich table of the differences between two dataframes.

    Parameters
    ----------
    df_old : pd.DataFrame
        The old dataframe to compare to.
    df_new : pd.DataFrame
        The new result dataframe.
    title : str
        Title for the comparison table.
    chunk_size : int | None
        Number of columns to show per line.
    """

    console = Console()

    df_old, df_new = df_old.align(df_new, join="outer", axis=None)
    diff_abs = df_new - df_old

    with np.errstate(divide="ignore", invalid="ignore"):
        diff_rel: pd.DataFrame = (df_new - df_old) / df_old

    all_columns = df_new.columns.tolist()
    if chunk_size is None or chunk_size <= 0:
        column_chunks = [all_columns]
    else:
        column_chunks = [
            all_columns[i : i + chunk_size]
            for i in range(0, len(all_columns), chunk_size)
        ]

    for i, cols in enumerate(column_chunks):
        current_title = title
        if len(column_chunks) > 1:
            current_title = f"{title} (Part {i + 1}/{len(column_chunks)})"

        table = Table(title=current_title, box=box.ROUNDED)

        table.add_column("Index", style="cyan", no_wrap=True)

        for col in cols:
            table.add_column(str(col), justify="right")

        for index, row in diff_abs.iterrows():
            assert isinstance(index, str)
            row_cells = [index]

            for col in cols:
                val_abs = row[col]
                assert isinstance(col, str)
                val_rel = diff_rel.at[index, col]
                assert isinstance(val_rel, float)

                if pd.isna(val_abs):
                    text_display = "-"
                else:
                    text_display = (
                        f"{val_abs:+3g}"
                        if isinstance(val_abs, (int, float))
                        else str(val_abs)
                    )

                style = ""

                if pd.isna(val_rel):
                    style = "dim"
                elif abs(val_rel) < 0.05 or abs(val_abs) < 1e-6:
                    style = "dim"  # Plain/Dim for no change
                elif val_rel >= 0.20:
                    # Strong Red for >= +20%
                    style = "bold white on red"
                elif val_rel > 0:
                    # Light Red/Salmon for positive but < 20%
                    style = "black on #ffcccb"
                elif val_rel <= -0.20:
                    # Strong Blue for <= -20%
                    style = "bold white on blue"
                elif val_rel < 0:
                    # Light Blue for negative but > -20%
                    style = "black on #add8e6"

                # Append the cell with style
                row_cells.append(f"[{style}]{text_display}[/{style}]")

            table.add_row(*row_cells)

        console.print(table)
        # Add a little spacing between chunks
        if i < len(column_chunks) - 1:
            console.print("")


@pytest.mark.parametrize(
    "resource_dir", BENCHMARK_CASES, ids=[d.stem for d in BENCHMARK_CASES]
)
def test_all_ims_benchmark_edge_cases(resource_dir: Path) -> None:
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
    print(np.ceil(np.log2(nt)))
    have_ko_matrix = np.ceil(np.log2(nt)) < 15
    if have_ko_matrix:
        im_list.append(ims.IM.FAS)

    # Calculate the intensity measures
    result = im_calculation.calculate_ims(
        waveform, dt, ims_list=im_list, ko_directory=KO_TEST_DIR
    )
    if not np.allclose(
        result.values,
        data.loc[result.index, result.columns].values,
        atol=5e-4,
        rtol=0.01,
    ):
        print_diff_table(
            data.loc[:, data.columns.str.startswith("pSA")],  # type: ignore[invalid-argument]
            result.loc[:, result.columns.str.startswith("pSA")],
            title=f"Differences for {resource_dir.stem}",
            chunk_size=8,
        )

    for im in result.columns:
        assert result[im].values == pytest.approx(
            data[im].values, abs=5e-4, rel=0.01, nan_ok=True
        ), (
            f"Results for {im} do not match!\n{result}"
        )  # 5e-6 implies rounding to five decimal places


@pytest.mark.parametrize("use_numexpr", [True, False])
def test_ds5xx(use_numexpr: bool) -> None:
    comp_0 = np.ones((100,), dtype=np.float32)
    waveforms = np.zeros((1, len(comp_0), 3), dtype=np.float32)
    waveforms[:, :, ims.Component.COMP_0] = comp_0
    # To stop invalid value errors when dividing by zero in other components
    waveforms[:, :, ims.Component.COMP_90] = comp_0 * 2
    waveforms[:, :, ims.Component.COMP_VER] = comp_0 * 3
    t_max = 1
    dt = t_max / len(comp_0)
    assert ims.ds575(waveforms, dt, use_numexpr)["000"].iloc[0] == pytest.approx(0.7)
    assert ims.ds595(waveforms, dt, use_numexpr)["000"].iloc[0] == pytest.approx(0.9)


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
) -> None:
    """Test peak ground motion parameter calculations."""
    dt = sample_time[1] - sample_time[0]

    if func == ims.peak_ground_acceleration:
        result = func(sample_waveforms)
    else:
        result = func(sample_waveforms, dt)

    # Check DataFrame structure
    assert isinstance(result, pd.DataFrame)
    columns = ["000", "090", "ver", "geom"]
    if func != ims.cumulative_absolute_velocity:
        columns.extend(["rotd0", "rotd50", "rotd100"])
    assert set(columns) == set(result.columns.tolist())
    # Check values
    assert np.all(result >= 0)  # All values should be non-negative


# Test cases for Arias Intensity
def test_arias_intensity(
    sample_waveforms: npt.NDArray[np.float32], sample_time: npt.NDArray[np.float32]
) -> None:
    """Test Arias Intensity calculation."""
    dt = sample_time[1] - sample_time[0]

    result = ims.arias_intensity(sample_waveforms, dt)

    # Check DataFrame structure
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns.tolist()) == {"000", "090", "ver", "geom"}
    # Check values
    assert np.all(result.select_dtypes(include=[np.number]) >= 0)
    # Check geom calculation
    assert_array_almost_equal(
        result["geom"].to_numpy(float), np.sqrt(result["000"] * result["090"])
    )


# Test cases for Fourier Amplitude Spectra
@pytest.mark.parametrize("n_freqs", [5, 10])
def test_fourier_amplitude_spectra(
    sample_waveforms: npt.NDArray[np.float32],
    sample_time: npt.NDArray[np.float32],
    n_freqs: int,
) -> None:
    """Test Fourier Amplitude Spectra calculation."""
    dt = sample_time[1] - sample_time[0]
    freqs = np.logspace(-1, 1, n_freqs, dtype=np.float32)
    # Force the multiprocessing code path if necessary.
    result_mp = ims.fourier_amplitude_spectra(
        sample_waveforms,
        dt,
        freqs,
        KO_TEST_DIR,
        cores=max(2, multiprocessing.cpu_count()),
    )
    # Force the single core path.
    result_sc = ims.fourier_amplitude_spectra(
        sample_waveforms, dt, freqs, KO_TEST_DIR, cores=1
    )

    # Check DataFrame structure
    assert isinstance(result_mp, xr.DataArray)
    assert list(result_mp.coords["component"]) == ["000", "090", "ver", "geom", "eas"]
    assert np.allclose(result_mp.coords["frequency"], freqs)
    assert np.all(result_mp.as_numpy() >= 0)
    # Check that multi-core result and single-core result produce the same output.
    assert np.allclose(result_mp.as_numpy(), result_sc.as_numpy())


def test_nyquist_frequency() -> None:
    # Define test parameters
    n_stations = 2
    n_timesteps = 1024
    n_components = 3
    dt = 0.01  # Timestep resolution (s)
    nyquist_frequency = 1 / (2 * dt)

    # Generate test waveforms (random data for simplicity)
    waveforms = np.random.rand(n_stations, n_timesteps, n_components).astype(np.float32)

    # Define frequencies, including some above the Nyquist frequency
    freqs = np.array(
        [1.0, 10.0, 20.0, 60.0], dtype=np.float32
    )  # 60 Hz > Nyquist (50 Hz)
    with pytest.warns(RuntimeWarning):
        fas = ims.fourier_amplitude_spectra(waveforms, dt, freqs, KO_TEST_DIR)

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
    waveforms = np.zeros(invalid_shape, dtype=np.float32)

    with pytest.raises((ValueError, IndexError)):
        ims.peak_ground_acceleration(waveforms)


# Edge cases
def test_zero_waveform() -> None:
    """Test behavior with zero-amplitude waveforms."""
    waveforms = np.zeros((2, 100, 3), dtype=np.float32)

    # Test various intensity measures with zero input
    pga_result = ims.peak_ground_acceleration(waveforms)

    # All results should be zero
    assert np.all(pga_result.select_dtypes(include=[np.number]) == 0)


@pytest.mark.parametrize("duration", [100, 200, 1000])
def test_numerical_stability(duration: int) -> None:
    """Test numerical stability with different duration lengths."""
    dt = 0.01
    t = np.arange(0, duration * dt, dt, dtype=np.float32)
    waveforms = np.zeros((2, len(t), 3), dtype=np.float32)
    # Add sine wave
    waveforms[:, :, ims.Component.COMP_0] = np.sin(2 * np.pi / (duration * dt) * t)

    result = ims.peak_ground_acceleration(waveforms)

    # Maximum should be close to 1.0 regardless of duration
    assert_array_almost_equal(result["000"].to_numpy(float), 1.0, decimal=5)
    # The others should just be 0.0
    assert_array_almost_equal(result["090"].to_numpy(float), 0.0, decimal=5)


@given(
    func=st.sampled_from(
        [
            ims.peak_ground_acceleration,
            ims.peak_ground_velocity,
        ]
    ),
    waveform=nst.arrays(
        np.float32,
        shape=st.tuples(st.integers(2, 10), st.integers(2, 10), st.just(3)),
        # Weird number is closest 32-bit floating point value to 0.01
        elements=st.floats(0.009999999776482582, 1, width=32).flatmap(
            lambda x: st.sampled_from([-1, 1]).flatmap(lambda sign: st.just(sign * x))
        ),
    ),
)
@settings(deadline=None)
def test_rotational_invariance(
    waveform: npt.NDArray[np.float32], func: Callable
) -> None:
    # DS595 and DS575 won't work if the waveform has values that are too small.
    assume(
        all(
            np.any(
                np.abs(
                    np.cos(np.radians(theta)) * waveform[:, :, 1]
                    + np.sin(np.radians(theta)) * waveform[:, :, 0]
                )
                > 1e-5,
                axis=1,
            ).all()
            for theta in range(180)
        )
    )
    if func != ims.peak_ground_acceleration:
        dt = 0.01
        func = functools.partial(func, dt=dt)
    old_waveform = np.copy(waveform)
    waveform_ims = func(waveform)
    assert np.allclose(old_waveform, waveform)
    waveform_ims_transposed = func(waveform[:, :, [1, 0, 2]])

    assert_array_almost_equal(
        waveform_ims["rotd0"], waveform_ims_transposed["rotd0"], decimal=3
    )
    assert_array_almost_equal(
        waveform_ims["rotd50"], waveform_ims_transposed["rotd50"], decimal=3
    )
    assert_array_almost_equal(
        waveform_ims["rotd100"], waveform_ims_transposed["rotd100"], decimal=3
    )


@given(
    waveform=nst.arrays(
        np.float32,
        shape=st.tuples(st.integers(2, 10), st.integers(2, 10), st.just(3)),
        elements=st.floats(-1, 1, width=32),
    ),
)
@settings(deadline=None)
def test_component_orientation(waveform: npt.NDArray[np.float32]) -> None:
    waveform_ims = ims.peak_ground_acceleration(waveform)

    assert_array_almost_equal(
        waveform_ims["000"].to_numpy(float),
        np.abs(waveform[:, :, ims.Component.COMP_0]).max(axis=1),
    )
    assert_array_almost_equal(
        waveform_ims["090"].to_numpy(float),
        np.abs(waveform[:, :, ims.Component.COMP_90]).max(axis=1),
    )
    assert_array_almost_equal(
        waveform_ims["ver"].to_numpy(float),
        np.abs(waveform[:, :, ims.Component.COMP_VER]).max(axis=1),
    )
