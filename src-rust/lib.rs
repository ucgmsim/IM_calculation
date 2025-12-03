mod psa;
mod rotd;
use pyo3::prelude::*;

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
mod _utils {
    use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
    use pyo3::prelude::*;

    use crate::psa;
    use crate::rotd;

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
}
