mod arias_intensity;
mod cav;
mod constants;
mod psa;
mod rotd;
mod significant_duration;
mod trapz;
mod utils;
use pyo3::prelude::*;

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
mod _utils {
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
        let waveform_psa = psa::newmark_beta_method_parallel(&waveforms, dt, w, xi);
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
    fn _parallel_arias_intensity<'py>(
        py: Python<'py>,
        waveforms_py: PyReadonlyArray2<f64>,
        dt: f64,
    ) -> Bound<'py, PyArray1<f64>> {
        let waveforms = waveforms_py.as_array();
        let waveform_ai = arias_intensity::parallel_arias_intensity(waveforms, dt);
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
    fn _parallel_cumulative_arias_intensity<'py>(
        py: Python<'py>,
        waveforms_py: PyReadonlyArray2<f64>,
        dt: f64,
    ) -> Bound<'py, PyArray2<f64>> {
        let waveforms = waveforms_py.as_array();
        let waveform_ai = arias_intensity::parallel_cumulative_arias_intensity(waveforms, dt);
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

    #[pyfunction]
    fn _parallel_cav<'py>(
        py: Python<'py>,
        waveforms_py: PyReadonlyArray2<f64>,
        dt: f64,
    ) -> Bound<'py, PyArray1<f64>> {
        let waveforms = waveforms_py.as_array();
        let waveform_cav = cav::parallel_cav(waveforms, dt);
        waveform_cav.into_pyarray(py)
    }

    #[pyfunction]
    fn _rotd_parallel<'py>(
        py: Python<'py>,
        comp_0_py: PyReadonlyArray2<f64>,
        comp_90_py: PyReadonlyArray2<f64>,
    ) -> Bound<'py, PyArray2<f64>> {
        let comp_0 = comp_0_py.as_array();
        let comp_90 = comp_90_py.as_array();
        let rotd_stats = rotd::rotd_parallel(comp_0, comp_90);
        rotd_stats.into_pyarray(py)
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
        let ds = significant_duration::significant_duration(arias_intensity, dt, low, high);
        ds.into_pyarray(py)
    }

    #[pyfunction]
    fn _parallel_significant_duration<'py>(
        py: Python<'py>,
        waveforms_py: PyReadonlyArray2<f64>,
        dt: f64,
        low: f64,
        high: f64,
    ) -> Bound<'py, PyArray1<f64>> {
        let waveforms = waveforms_py.as_array();
        let arias_intensity = arias_intensity::parallel_cumulative_arias_intensity(waveforms, dt);
        let ds =
            significant_duration::parallel_significant_duration(arias_intensity, dt, low, high);
        ds.into_pyarray(py)
    }
}
