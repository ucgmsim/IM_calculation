from pathlib import Path

import pint_xarray  # noqa: F401
import xarray as xr

from IM.ims import IM


def read_intensity_measures(
    intensity_measure_file: str | Path,
) -> xr.Dataset:
    """
    Read intensity measures from a file and return as an xarray Dataset.

    Parameters
    ----------
    intensity_measure_file : str or Path
        The file path or filename of the intensity measures dataset to be read.

    Returns
    -------
    xr.Dataset
        The xarray dataset containing the intensity measures."""
    return xr.open_dataset(intensity_measure_file, engine="h5netcdf").pint.quantify()


COORDINATE_METADATA = {
    "station": {"description": "Station identifiers", "units": None},
    "component": {"description": "Component of motion", "units": None},
    "period": {"description": "Oscillation period", "units": "s"},
    "vs30": {
        "description": "Average shear-wave velocity to 30m depth",
        "units": "m/s",
    },
    "epi": {"description": "Epicentral distance", "units": "km"},
    "hyp": {"description": "Hypocentral distance", "units": "km"},
    "rrup": {"description": "Rupture distance", "units": "km"},
    "rjb": {"description": "Joyner-Boore distance", "units": "km"},
    "latitude": {"description": "Station latitude", "units": "degrees"},
    "longitude": {"description": "Station longitude", "units": "degrees"},
    "frequency": {"description": "Frequency of motion", "units": "Hz"},
}

IM_METADATA = {
    IM.PGA: "Peak ground acceleration",
    IM.PGV: "Peak ground velocity",
    IM.CAV: "Cumulative absolute velocity",
    IM.AI: "Arias intensity",
    IM.Ds575: "Significant duration (5-75%)",
    IM.Ds595: "Significant duration (5-95%)",
    IM.pSA: "Pseudo-spectral acceleration",
    IM.FAS: "Fourier amplitude spectrum",
}


IM_UNITS = {
    IM.PGA: "g0",
    IM.PGV: "cm/s",
    IM.CAV: "m/s",
    IM.CAV5: "m/s",
    IM.AI: "m/s",
    IM.Ds575: "s",
    IM.Ds595: "s",
    IM.FAS: "g0 * s",
    IM.pSA: "g0",
    "frequency": "Hz",
    "period": "s",
}


def write_intensity_measures(dataset: xr.Dataset, output_ffp: str | Path) -> None:
    """Write intensity measures to a file, updating coordinate and variable metadata.

    Parameters
    ----------
    dataset : xr.Dataset
        The xarray dataset containing intensity measures to be written.
    output_ffp : str or Path
        The file path where the output dataset should be saved.
    """
    for name, description in COORDINATE_METADATA.items():
        if name not in dataset.coords:
            continue
        dataset.coords[name].attrs.update(COORDINATE_METADATA[name])

    for im_name, description in IM_METADATA.items():
        if im_name not in dataset:
            continue
        dataset[im_name].attrs["description"] = IM_METADATA[im_name]

    dataset = dataset.pint.quantify(IM_UNITS)

    dataset.pint.dequantify().to_netcdf(output_ffp, engine="h5netcdf")
