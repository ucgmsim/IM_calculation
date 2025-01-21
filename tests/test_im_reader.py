from pathlib import Path

import pandas as pd
import pytest
import xarray as xr

from IM.im_reader import IMFile
from IM.ims import IM


@pytest.fixture
def hdf5_file(tmp_path: Path):
    """Fixture to create a temporary HDF5 file with test data for all IMs."""
    file_path = tmp_path / "test_file.h5"

    # Write data using xarray and pandas
    for im in IM:
        if im in IMFile._xarray_methods:
            data_array = xr.DataArray([1.0, 2.0, 3.0], dims=["dim_0"])
            data_array.to_netcdf(file_path, mode="a", group=im, engine="h5netcdf")
        else:
            df = pd.DataFrame({"index": ["A", "B", "C"], "values": [10, 20, 30]})
            df.to_hdf(file_path, key=im, mode="a")

    return file_path

@pytest.mark.parametrize("intensity_measure", IM)
def test_getitem_all_ims(hdf5_file: Path, intensity_measure: IM):
    """Test reading all intensity measures from disk."""
    im_file = IMFile(hdf5_file)

    if intensity_measure in IMFile._xarray_methods:
        data = im_file[intensity_measure]
        assert isinstance(data, xr.DataArray)
        assert data.shape == (3,)
        assert list(data.values) == [1.0, 2.0, 3.0]
    else:
        data = im_file[intensity_measure]
        assert isinstance(data, pd.DataFrame)
        assert list(data["values"]) == [10, 20, 30]
        assert list(data["index"]) == ["A", "B", "C"]

def test_initialization(hdf5_file: Path):
    """Test initialization of the IMFile object."""
    im_file = IMFile(hdf5_file)
    assert im_file.path == hdf5_file
