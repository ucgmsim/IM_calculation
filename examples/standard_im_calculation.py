from pathlib import Path

import numpy as np

from IM import im_calculation, waveform_reading
from IM.im_calculation import IM

data_dir = Path(__file__).parent / "resources"

comp_000_ffp = data_dir / "DSZ.000"
comp_090_ffp = data_dir / "DSZ.090"
comp_ver_ffp = data_dir / "DSZ.ver"

dt, waveform = waveform_reading.read_ascii(comp_000_ffp, comp_090_ffp, comp_ver_ffp)

# Reshape the waveform to have the correct shape for the IM calculation
reshaped_waveform = waveform[np.newaxis, :, :]

im_results = im_calculation.calculate_ims(
    reshaped_waveform, dt, [IM.PGA], ["000", "090", "ver", "rotd0", "rotd50", "rotd100"]
)

print(im_results)