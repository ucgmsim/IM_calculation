from pathlib import Path

import pandas as pd

from IM import im_calculation, waveform_reading

data_dir = Path(__file__).parent / "resources"

comp_000_ffp = data_dir / "2024p950420_MWFS_HN_20.000"
comp_090_ffp = data_dir / "2024p950420_MWFS_HN_20.090"
comp_ver_ffp = data_dir / "2024p950420_MWFS_HN_20.ver"

# Read the files to a waveform array that's readable by IM Calculation
dt, waveform = waveform_reading.read_ascii(comp_000_ffp, comp_090_ffp, comp_ver_ffp)

# Calculate the intensity measures using defaults
im_results = im_calculation.calculate_ims(waveform, dt)

# Sanity check that the IMs that are calculated are the ones we expect
# By comparing against the benchmark set
benchmark_dir = Path(__file__).parent.parent / "tests" / "resources"
benchmark_im_results = pd.read_csv(benchmark_dir / "im_benchmark.csv", index_col=0)

# The benchmark predates the RotD orientation components, so compare only the
# components it has.
components = [c for c in im_results.index if c in benchmark_im_results.index]
pd.testing.assert_frame_equal(
    im_results.loc[components],
    benchmark_im_results.loc[components],
    atol=5e-4,
    rtol=0.01,
)
