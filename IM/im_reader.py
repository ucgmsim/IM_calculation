"""Utility module for reading intensity measure files stored in HDF5 format."""

from pathlib import Path

import pandas as pd
import xarray as xr

from IM.ims import IM


class IMFile:
    """
    A class for reading intensity measure (IM) data from an HDF5 file.

    This class provides methods to retrieve IM data as either a
    `pandas.DataFrame` or an `xarray.DataArray`, depending on the type
    of intensity measure requested.

    Attributes
    ----------
    path : Path
        The file path to the HDF5 file containing intensity measure data.
    _xarray_methods : set
        A set of intensity measures that should be accessed as
        `xarray.DataArray` objects.
    """

    _xarray_methods = {IM.pSA, IM.FAS}

    def __init__(self, path: Path):
        """
        Initialize the IMFile instance.

        Parameters
        ----------
        path : Path
            The file path to the HDF5 file containing intensity measure data.
        """
        self.path = path

    def __getitem__(self, intensity_measure: IM) -> pd.DataFrame | xr.DataArray:
        """Retrieve data for a given intensity measure.

        Parameters
        ----------
        intensity_measure : IM
            The intensity measure for which data is to be retrieved. This
            must match a key or group in the HDF5 file.

        Returns
        -------
        pd.DataFrame or xr.DataArray
            The data associated with the given intensity measure. Returns
            an `xarray.DataArray` if the intensity measure is in
            `_xarray_methods`, otherwise a `pandas.DataFrame`.

        Raises
        ------
        AttributeError
            If the intensity measure is not found or is invalid.

        Examples
        --------
        >>> from pathlib import Path
        >>> im_reader = IMFile(Path("path/to/file.h5"))
        >>> data = im_reader.__getattribute__(IM.pSA)
        >>> type(data)
        <class 'xarray.DataArray'>  # If IM.pSA is in `_xarray_methods`
        """
        if intensity_measure in self._xarray_methods:
            return xr.open_dataarray(
                self.path, engine="h5netcdf", group=intensity_measure
            )
        else:
            return pd.read_hdf(self.path, key=intensity_measure)
