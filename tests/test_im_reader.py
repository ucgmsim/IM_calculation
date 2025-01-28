from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from IM import ims
from IM.im_reader import (
    COORDINATE_METADATA,
    IM_METADATA,
    IM_UNITS,
    read_intensity_measures,
    write_intensity_measures,
)


@pytest.fixture
def sample_dataset() -> xr.Dataset:
    """Fixture to create a sample xarray dataset for testing."""
    data = np.random.rand(7, 3, 10)
    coords = {
        "station": ["A", "B", "C"],
        "component": ["000", "090", "ver", "geom", "rotd0", "rotd50", "rotd100"],
        "period": np.linspace(0.01, 1.0, 10),
    }
    variables = {
        "pSA": (["component", "station", "period"], data),
        "PGA": (["component", "station"], np.random.rand(7, 3)),
        "PGV": (["component", "station"], np.random.rand(7, 3)),
        "CAV": (["component", "station"], np.random.rand(7, 3)),
        "CAV5": (["component", "station"], np.random.rand(7, 3)),
        "AI": (["component", "station"], np.random.rand(7, 3)),
        "Ds575": (["component", "station"], np.random.rand(7, 3)),
        "Ds595": (["component", "station"], np.random.rand(7, 3)),
        "FAS": (["component", "station", "period"], np.random.rand(7, 3, 10)),
    }
    ds = xr.Dataset(variables, coords=coords)

    # Add units to the variables
    for var, units in IM_UNITS.items():
        if isinstance(var, ims.IM):
            ds[var].attrs["units"] = units

    return ds


def test_read_intensity_measures_with_units(tmp_path: Path, sample_dataset: xr.Dataset):
    """Test reading intensity measures with units enabled."""
    file_path = tmp_path / "test_file.h5"
    sample_dataset.to_netcdf(file_path, engine="h5netcdf")

    result = read_intensity_measures(file_path)

    assert isinstance(result, xr.Dataset)
    for im_name in IM_METADATA.keys():
        assert im_name in result
        assert hasattr(result[im_name], "pint")
        assert result[im_name].pint.units is not None


def test_write_intensity_measures(tmp_path: Path, sample_dataset: xr.Dataset):
    """Test writing intensity measures with metadata updates."""
    file_path = tmp_path / "output_file.h5"

    write_intensity_measures(sample_dataset, file_path)

    # Reload the written dataset
    result = xr.open_dataset(file_path, engine="h5netcdf")

    # Check coordinate metadata
    for coord, metadata in COORDINATE_METADATA.items():
        if coord in result.coords:
            assert result.coords[coord].attrs["description"] == metadata["description"]

    # Check variable metadata
    for im_name, description in IM_METADATA.items():
        if im_name in result:
            assert result[im_name].attrs["description"] == description

    # Ensure units were applied and then dequantified
    for im_name in IM_METADATA.keys():
        if im_name in result:
            assert "units" in result[im_name].attrs


def test_write_intensity_measures_missing_metadata(
    tmp_path: Path, sample_dataset: xr.Dataset
):
    """Test writing intensity measures with missing metadata in dataset."""
    # Remove some metadata to simulate missing attributes

    file_path = tmp_path / "output_file.h5"

    write_intensity_measures(sample_dataset, file_path)

    result = xr.open_dataset(file_path, engine="h5netcdf")

    # Ensure that missing metadata does not cause an error
    assert "station" in result.coords
    assert "description" in result.coords["station"].attrs

    for im_name in IM_METADATA.keys():
        if im_name in result:
            assert "description" in result[im_name].attrs
