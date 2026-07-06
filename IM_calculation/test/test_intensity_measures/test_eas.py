"""Self-contained tests for the NGA-West2 EAS port (no obspy / BB fixtures)."""
import numpy as np
import pytest
from scipy.interpolate import interp1d

from IM_calculation.IM import computeFAS

DT = 0.005
NT = 64                       # nfft 64 -> rfft length 33 -> KO_32.npy (33x33)
OUT_FREQS = np.array([1.0, 5.0, 20.0, 50.0])


def _toy_konno(directory, size, seed=0):
    """Write a deterministic column-normalised (size, size) smoothing matrix."""
    rng = np.random.default_rng(seed)
    m = rng.random((size, size)) + 0.1
    m /= m.sum(axis=0, keepdims=True)
    np.save(directory / f"KO_{size - 1}.npy", m)
    return m


@pytest.fixture(autouse=True)
def _clear_konno_cache():
    computeFAS.matrices.clear()
    yield
    computeFAS.matrices.clear()


@pytest.fixture
def konno(tmp_path):
    return tmp_path, _toy_konno(tmp_path, 33)     # rfft length for NT=64 is 33


def _waveform(comp0, comp1, comp2):
    w = np.zeros((NT, 3))
    w[:, 0], w[:, 1], w[:, 2] = comp0, comp1, comp2
    return w


def test_eas_equals_single_component_fas_when_horizontals_identical(konno):
    ko_dir, _ = konno
    t = np.arange(NT) * DT
    h = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 30 * t)
    w = _waveform(h, h.copy(), np.cos(2 * np.pi * 15 * t))     # 090 == 000

    fas, eas = computeFAS.get_fourier_spectrum_with_eas(
        w, DT, OUT_FREQS, ko_directory=ko_dir
    )

    assert fas.shape == (len(OUT_FREQS), 3)                    # includes vertical
    # sqrt(0.5*(x^2 + x^2)) == x, so smoothed EAS == smoothed single-component FAS
    assert np.allclose(eas, fas[:, 0])
    # the old euclidean definition would be sqrt(2) larger -> must NOT match
    assert not np.allclose(eas, np.sqrt(2.0) * fas[:, 0])


def test_eas_and_fas_match_independent_reference(konno):
    ko_dir, m = konno
    t = np.arange(NT) * DT
    w = _waveform(
        np.sin(2 * np.pi * 8 * t),
        np.sin(2 * np.pi * 12 * t + 0.3),
        np.cos(2 * np.pi * 20 * t),
    )

    fas, eas = computeFAS.get_fourier_spectrum_with_eas(
        w, DT, OUT_FREQS, ko_directory=ko_dir
    )

    raw = np.abs(np.fft.rfft(w, n=NT, axis=0) * DT)            # (33, 3)
    fa_freq = np.fft.rfftfreq(NT, DT)
    fas_ref = interp1d(fa_freq, np.dot(raw.T, m).T, axis=0,
                       fill_value="extrapolate")(OUT_FREQS)
    eas_unsm = np.sqrt(0.5 * (raw[:, 0] ** 2 + raw[:, 1] ** 2))
    eas_ref = interp1d(fa_freq, np.dot(eas_unsm, m),
                       fill_value="extrapolate")(OUT_FREQS)

    assert np.allclose(fas, fas_ref)
    assert np.allclose(eas, eas_ref)
