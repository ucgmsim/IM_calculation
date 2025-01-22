from pathlib import Path

import pytest
from typer.testing import CliRunner

from IM.scripts import calculate_ims
from IM.scripts.calculate_ims import (
    app,  # Replace with the actual module name containing your app
)
import shutil

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
    assert "Usage:" in result.stdout


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
    assert "Usage:" in result.stdout
