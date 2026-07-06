"""Small pure-numpy helpers for pSA/SDI period handling."""
import numpy as np


def collapse_fp_duplicates(periods, canonical, rtol=1e-9):
    """Merge floating-point near-duplicate periods onto their canonical value.

    ``np.logspace`` computes a whole power of ten (e.g. ``10 ** -1``) as the adjacent
    double on some platforms (``0.09999999999999999`` instead of ``0.1``). Appended to
    the requested periods and passed through ``np.unique``, that stray double survives
    as a *separate* period and the output grows a spurious ``pSA_0.09999999999999999``
    column alongside ``pSA_0.1``. Snapping any period that is numerically equal to a
    requested (canonical) period onto that exact value collapses the duplicate while
    leaving genuinely distinct periods untouched, so every other column label is
    preserved byte-for-byte.

    Parameters
    ----------
    periods : array-like of float
        Candidate periods, possibly containing floating-point near-duplicates.
    canonical : array-like of float
        Reference periods to snap onto (the user/DEFAULT requested periods).
    rtol : float, optional
        Relative tolerance for treating two periods as the same. The default 1e-9 sits
        far below the ~0.5% minimum gap between real periods and far above ~1e-16
        floating-point noise.

    Returns
    -------
    np.ndarray
        Sorted, de-duplicated periods with floating-point duplicates merged onto
        ``canonical``.
    """
    periods = np.array(periods, dtype="float64")
    for c in np.array(canonical, dtype="float64"):
        periods[np.isclose(periods, c, rtol=rtol, atol=0.0)] = c
    return np.unique(periods)
