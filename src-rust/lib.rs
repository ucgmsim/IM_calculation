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
    use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
    use pyo3::prelude::*;

    use crate::arias_intensity;
    use crate::cav;
    use crate::psa;
    use crate::rotd;
    use crate::significant_duration;

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

    /// Pseudo-spectral acceleration statistics for every station and period.
    ///
    /// The three components have shape `(ns, nt)`. Returns an `(ns,
    /// n_periods, 10)` array whose last axis is laid out as
    /// `ROTD_COMPONENTS`: the 000, 090, vertical and geometric mean peaks,
    /// then the six columns [`_rotd`] returns.
    #[pyfunction]
    fn _psa<'py>(
        py: Python<'py>,
        comp_0_py: PyReadonlyArray2<f64>,
        comp_90_py: PyReadonlyArray2<f64>,
        comp_ver_py: PyReadonlyArray2<f64>,
        periods_py: PyReadonlyArray1<f64>,
        dt: f64,
        xi: f64,
    ) -> Bound<'py, PyArray3<f64>> {
        let comp_0 = comp_0_py.as_array();
        let comp_90 = comp_90_py.as_array();
        let comp_ver = comp_ver_py.as_array();
        let periods = periods_py.as_array();
        let psa = py.detach(|| psa::psa(&comp_0, &comp_90, &comp_ver, &periods, dt, xi));
        psa.into_pyarray(py)
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
