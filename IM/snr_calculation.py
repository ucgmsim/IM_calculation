import numpy as np

from IM import ims, im_calculation


def calculate_snr(
    waveform: np.ndarray,
    dt: float,
    tp: float,
    frequencies: np.asarray = im_calculation.DEFAULT_FREQUENCIES,
):
    """
    Calculates the SNR of a waveform given a tp and common frequency vector

    Parameters
    ----------
    waveform : np.ndarray
        Waveform data as a NumPy array.
    dt : float
        The sampling rate of the waveform
    tp : float
        The index of the p-arrival
    frequencies : np.asarray, optional
        The frequency vector to use for the SNR calculation,
        by default takes the frequencies from FAS
    """
    # Calculate signal and noise areas
    signal_acc, noise_acc = waveform[:, tp:], waveform[:, :tp]
    signal_duration, noise_duration = len(signal_acc) * dt, len(noise_acc) * dt

    # Ensure the noise is not shorter than 1s, if not then skip the calculation
    if noise_duration < 1:
        return None, None, None, None, None, None

    # Add the tapering to the signal and noise
    taper_signal_acc = apply_taper(signal_acc)
    taper_noise_acc = apply_taper(noise_acc)

    # Generate FFT for the signal and noise
    fas_signal, frequency_signal = computeFAS.generate_fa_spectrum(
        taper_signal_acc, waveform.DT, len(taper_signal_acc)
    )
    fas_noise, frequency_noise = computeFAS.generate_fa_spectrum(
        taper_noise_acc, waveform.DT, len(taper_signal_acc)
    )

    # Take the absolute value of the FAS
    fas_signal = np.abs(fas_signal)
    fas_noise = np.abs(fas_noise)

    if frequencies is not None:
        # Interpolate FAS at common frequencies
        inter_signal_f = interp1d(
            frequency_signal, fa_smooth_signal, axis=0, bounds_error=False
        )
        inter_noise_f = interp1d(
            frequency_noise, fa_smooth_noise, axis=0, bounds_error=False
        )
        inter_signal = inter_signal_f(frequencies)
        inter_noise = inter_noise_f(frequencies)
    else:
        inter_signal = fa_smooth_signal
        inter_noise = fa_smooth_noise

    # Set values to NaN if they are outside the bounds of sample rate / 2
    inter_signal[frequencies > sampling_rate / 2] = np.nan
    inter_noise[frequencies > sampling_rate / 2] = np.nan

    # Calculate the SNR
    with np.errstate(divide='ignore', invalid='ignore'):
        snr = (inter_signal / np.sqrt(signal_duration)) / (
            inter_noise / np.sqrt(noise_duration)
        )
    frequencies = (
        frequency_signal if frequencies is None else frequencies
    )

    return snr, frequencies, inter_signal, inter_noise, signal_duration, noise_duration