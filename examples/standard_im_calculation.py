from pathlib import Path

from IM import im_calculation, waveform_reading

data_dir = Path(__file__).parent / "resources"

comp_000_ffp = data_dir / "2024p950420_MWFS_HN_20.000"
comp_090_ffp = data_dir / "2024p950420_MWFS_HN_20.090"
comp_ver_ffp = data_dir / "2024p950420_MWFS_HN_20.ver"

# Read the files to a waveform array that's readable by IM Calculation
dt, waveform = waveform_reading.read_ascii(comp_000_ffp, comp_090_ffp, comp_ver_ffp)

# Calculate the intensity measures using defaults
im_results = im_calculation.calculate_ims(
    waveform, dt
)

print(im_results)