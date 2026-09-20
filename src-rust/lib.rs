pub mod arias_intensity;
pub mod cav;
pub mod constants;
pub mod psa;
pub mod rotd;
pub mod significant_duration;
mod trapz;
use pyo3::prelude::*;

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
mod _core {
    use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
    use pyo3::prelude::*;

    use crate::arias_intensity;
    use crate::cav;
    use crate::psa;
    use crate::rotd;
    use crate::significant_duration;

    /// Newmark-beta method
    #[pyfunction]
    fn _newmark_beta_method<'py>(
        py: Python<'py>,
        waveforms_py: PyReadonlyArray2<f64>,
        dt: f64,
        w: f64,
        xi: f64,
    ) -> Bound<'py, PyArray2<f64>> {
        let waveforms = waveforms_py.as_array();
        let waveform_psa = py.detach(|| psa::newmark_beta_method_batch(&waveforms, dt, w, xi));
        waveform_psa.into_pyarray(py)
    }

    #[pyfunction]
    fn _arias_intensity<'py>(
        py: Python<'py>,
        waveforms_py: PyReadonlyArray2<f64>,
        dt: f64,
    ) -> Bound<'py, PyArray1<f64>> {
        let waveforms = waveforms_py.as_array();
        let waveform_ai = py.detach(|| arias_intensity::arias_intensity(waveforms, dt));
        waveform_ai.into_pyarray(py)
    }

    #[pyfunction]
    fn _cav<'py>(
        py: Python<'py>,
        waveforms_py: PyReadonlyArray2<f64>,
        dt: f64,
    ) -> Bound<'py, PyArray1<f64>> {
        let waveforms = waveforms_py.as_array();
        let waveform_cav = py.detach(|| cav::cav(waveforms, dt));
        waveform_cav.into_pyarray(py)
    }

    /// Serial pSA at all 180 rotation angles for one period.
    #[pyfunction]
    fn _psa_rotd180<'py>(
        py: Python<'py>,
        comp_0_py: PyReadonlyArray2<f64>,
        comp_90_py: PyReadonlyArray2<f64>,
        dt: f64,
        w: f64,
        xi: f64,
    ) -> Bound<'py, PyArray2<f64>> {
        let comp_0 = comp_0_py.as_array();
        let comp_90 = comp_90_py.as_array();
        let psa_rotd = py.detach(|| psa::psa_rotd180(&comp_0, &comp_90, dt, w, xi));
        psa_rotd.into_pyarray(py)
    }

    /// Pseudo-spectral acceleration peak for a single component, one period.
    #[pyfunction]
    fn _psa_peak<'py>(
        py: Python<'py>,
        waveforms_py: PyReadonlyArray2<f64>,
        dt: f64,
        w: f64,
        xi: f64,
    ) -> Bound<'py, PyArray1<f64>> {
        let waveforms = waveforms_py.as_array();
        let peak = py.detach(|| psa::psa_peak(&waveforms, dt, w, xi));
        peak.into_pyarray(py)
    }

    /// RotD statistics of every `(comp_0, comp_90)` pair.
    ///
    /// Returns an `(ns, 6)` array: RotD00, RotD50 and RotD100, then the
    /// orientation in degrees at which each of the three occurs.
    #[pyfunction]
    fn _rotd<'py>(
        py: Python<'py>,
        comp_0_py: PyReadonlyArray2<f64>,
        comp_90_py: PyReadonlyArray2<f64>,
    ) -> Bound<'py, PyArray2<f64>> {
        let comp_0 = comp_0_py.as_array();
        let comp_90 = comp_90_py.as_array();
        let rotd_stats = py.detach(|| rotd::rotd(comp_0, comp_90));
        rotd_stats.into_pyarray(py)
    }

    /// The same RotD statistics, reduced from an already computed angle sweep.
    ///
    /// `curve_py` is an `(ns, 180)` array of peaks at every integer angle, as
    /// the first 180 columns of [`_psa_rotd180`]. Returns the `(ns, 6)` array
    /// [`_rotd`] returns, so the pSA path shares one reduction with the peak
    /// ground motion path instead of repeating it in numpy.
    #[pyfunction]
    fn _rotd180_stats<'py>(
        py: Python<'py>,
        curve_py: PyReadonlyArray2<f64>,
    ) -> Bound<'py, PyArray2<f64>> {
        let curve = curve_py.as_array();
        let rotd_stats = py.detach(|| rotd::rotd180_stats(curve));
        rotd_stats.into_pyarray(py)
    }

    #[pyfunction]
    fn _significant_duration<'py>(
        py: Python<'py>,
        waveforms_py: PyReadonlyArray2<f64>,
        dt: f64,
        low: f64,
        high: f64,
    ) -> Bound<'py, PyArray1<f64>> {
        let waveforms = waveforms_py.as_array();
        let ds = py.detach(|| {
            let arias_intensity = arias_intensity::cumulative_arias_intensity(waveforms, dt);
            significant_duration::significant_duration(arias_intensity.view(), dt, low, high)
        });
        ds.into_pyarray(py)
    }
}
