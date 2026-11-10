import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from IM import im_calculation, waveform_reading
from IM.scripts.calculate_ims import (
    app,  # Replace with the actual module name containing your app
)

runner = CliRunner()


@pytest.fixture
def waveform_directory(tmp_path: Path) -> Path:
    shutil.copytree(
        Path(__file__).parent.parent / "examples" / "resources",
        tmp_path,
        dirs_exist_ok=True,
    )
    return tmp_path


def test_calculate_ims_ascii_success(waveform_directory: Path):
    # Setup: Create temporary files for testing
    file_000 = waveform_directory / "2024p950420_MWFS_HN_20.000"
    file_090 = waveform_directory / "2024p950420_MWFS_HN_20.090"
    file_ver = waveform_directory / "2024p950420_MWFS_HN_20.ver"
    output_file = waveform_directory / "output.csv"

    ims_list = ["PGA", "PGV"]

    # Run the command
    result = runner.invoke(
        app,
        [
            str(file_000),
            str(file_090),
            str(file_ver),
            str(output_file),
        ]
        + ims_list,
    )

    # Assertions
    assert result.exit_code == 0
    assert output_file.exists()


def test_calculate_ims_ascii_invalid_file():
    # Test with a non-existent file
    result = runner.invoke(
        app,
        [
            "nonexistent_000.txt",
            "nonexistent_090.txt",
            "nonexistent_ver.txt",
            "output.csv",
            "PGA",
        ],
    )

    # Assertions
    assert result.exit_code != 0
    assert "Usage:" in result.stderr


def test_calculate_ims_ascii_matches_calculate_ims(waveform_directory: Path):
    file_000 = waveform_directory / "2024p950420_MWFS_HN_20.000"
    file_090 = waveform_directory / "2024p950420_MWFS_HN_20.090"
    file_ver = waveform_directory / "2024p950420_MWFS_HN_20.ver"
    output_file = waveform_directory / "output.csv"

    result = runner.invoke(
        app,
        [
            str(file_000),
            str(file_090),
            str(file_ver),
            str(output_file),
            "pSA",
            "--periods",
            "0.5",
            "--periods",
            "1.0",
        ],
    )
    assert result.exit_code == 0, result.output

    dt, waveform = waveform_reading.read_ascii(file_000, file_090, file_ver)
    expected = im_calculation.calculate_ims(
        waveform, dt, [im_calculation.IM.pSA], np.array([0.5, 1.0])
    )
    actual = pd.read_csv(output_file, index_col=0)

    assert list(actual.columns) == list(expected.columns)
    assert list(actual.index) == list(expected.index)
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), rtol=1e-12)


def test_calculate_ims_ascii_empty_ims_list(tmp_path: Path):
    # Setup: Create temporary files for testing
    file_000 = tmp_path / "000.txt"
    file_090 = tmp_path / "090.txt"
    file_ver = tmp_path / "ver.txt"
    output_file = tmp_path / "output.csv"

    # Write dummy data to the input files
    file_000.write_text("0.1\n0.2\n0.3")
    file_090.write_text("0.1\n0.2\n0.3")
    file_ver.write_text("0.1\n0.2\n0.3")

    # Run the command with an empty IMS list
    result = runner.invoke(
        app,
        [
            str(file_000),
            str(file_090),
            str(file_ver),
            str(output_file),
        ],
    )

    # Assertions
    assert result.exit_code != 0
    assert "Usage:" in result.stderr
