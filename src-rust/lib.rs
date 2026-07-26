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
        let waveform_psa = psa::newmark_beta_method_batch(&waveforms, dt, w, xi);
        waveform_psa.into_pyarray(py)
    }

    #[pyfunction]
    fn _arias_intensity<'py>(
        py: Python<'py>,
        waveforms_py: PyReadonlyArray2<f64>,
        dt: f64,
    ) -> Bound<'py, PyArray1<f64>> {
        let waveforms = waveforms_py.as_array();
        let waveform_ai = arias_intensity::arias_intensity(waveforms, dt);
        waveform_ai.into_pyarray(py)
    }

    #[pyfunction]
    fn _cumulative_arias_intensity<'py>(
        py: Python<'py>,
        waveforms_py: PyReadonlyArray2<f64>,
        dt: f64,
    ) -> Bound<'py, PyArray2<f64>> {
        let waveforms = waveforms_py.as_array();
        let waveform_ai = arias_intensity::cumulative_arias_intensity(waveforms, dt);
        waveform_ai.into_pyarray(py)
    }

    #[pyfunction]
    fn _cav<'py>(
        py: Python<'py>,
        waveforms_py: PyReadonlyArray2<f64>,
        dt: f64,
    ) -> Bound<'py, PyArray1<f64>> {
        let waveforms = waveforms_py.as_array();
        let waveform_cav = cav::cav(waveforms, dt);
        waveform_cav.into_pyarray(py)
    }

    /// Serial pSA at all 180 rotation angles for one period.
    ///
    /// Runs the Newmark-beta solver (f64) and the RotD reduction entirely in
    /// Rust, one station after another, so a Dask worker holding a single core
    /// gets no competing Rayon threads. Returns an `(ns, 180)` array of pSA.
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
        // The solve touches no Python objects, so drop the GIL for its whole
        // duration: a threaded caller (e.g. Dask's threaded scheduler) can then
        // run one of these per core in parallel within a single process.
        let psa_rotd = py.detach(|| psa::psa_rotd180(&comp_0, &comp_90, dt, w, xi));
        psa_rotd.into_pyarray(py)
    }

    #[pyfunction]
    fn _rotd<'py>(
        py: Python<'py>,
        comp_0_py: PyReadonlyArray2<f64>,
        comp_90_py: PyReadonlyArray2<f64>,
    ) -> Bound<'py, PyArray2<f64>> {
        let comp_0 = comp_0_py.as_array();
        let comp_90 = comp_90_py.as_array();
        let rotd_stats = rotd::rotd(comp_0, comp_90);
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
        let arias_intensity = arias_intensity::cumulative_arias_intensity(waveforms, dt);
        let ds = significant_duration::significant_duration(arias_intensity.view(), dt, low, high);
        ds.into_pyarray(py)
    }
}
