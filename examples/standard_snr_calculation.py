from pathlib import Path

from IM import snr_calculation, waveform_reading

data_dir = Path(__file__).parent / "resources"

comp_000_ffp = data_dir / "2024p950420_MWFS_HN_20.000"
comp_090_ffp = data_dir / "2024p950420_MWFS_HN_20.090"
comp_ver_ffp = data_dir / "2024p950420_MWFS_HN_20.ver"

# Index of the start of the P-wave
tp = 3170

# Read the files to a waveform array that's readable by IM Calculation
dt, waveform = waveform_reading.read_ascii(comp_000_ffp, comp_090_ffp, comp_ver_ffp)

# Calculate the SNR using defaults
snr_results = snr_calculation.calculate_snr(waveform, dt, tp)

print(snr_results)
