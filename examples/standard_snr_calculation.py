from pathlib import Path

from IM import snr_calculation, waveform_reading

data_dir = Path(__file__).parent / "resources"

comp_000_ffp = data_dir / "DSZ.000"
comp_090_ffp = data_dir / "DSZ.090"
comp_ver_ffp = data_dir / "DSZ.ver"

# Index of the start of the P-wave
tp = 1320

# Read the files to a waveform array that's readable by IM Calculation
dt, waveform = waveform_reading.read_ascii(comp_000_ffp, comp_090_ffp, comp_ver_ffp)

# Calculate the SNR using defaults
snr_results = snr_calculation.calculate_snr(
    waveform, dt, tp
)

print(snr_results)