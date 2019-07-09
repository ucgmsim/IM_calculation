import argparse
import glob
import os
import pandas as pd
import subprocess

DEFAULT_OPEN_SEES_PATH = "OpenSees"
model_dir = os.path.dirname(__file__)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "comp_090", help="filepath to a station's 090 waveform ascii file"
    )
    parser.add_argument(
        "comp_000", help="filepath to a station's 000 waveform ascii file"
    )
    parser.add_argument(
        "comp_ver", help="filepath to a station's ver waveform ascii file"
    )
    parser.add_argument(
        "output_dir",
        help="Where the IM_csv file is written to. Also contains the temporary recorders output",
    )
    parser.add_argument(
        "--OpenSees_path",
        default=DEFAULT_OPEN_SEES_PATH,
        help="Path to OpenSees binary",
    )

    args = parser.parse_args()

    script = [
        args.OpenSees_path,
        os.path.join(model_dir, "run_script.tcl"),
        args.comp_090,
        args.comp_000,
        args.comp_ver,
        args.output_dir,
    ]

    print(" ".join(script))
    subprocess.call(script)

    create_im_csv(args.output_dir, "Steel_MF_5Story")


def create_im_csv(output_dir, im_name):
    im_csv_fname = os.path.join(output_dir, im_name + ".csv")

    stations_dir = os.path.join(output_dir, "*")
    stations = glob.glob(stations_dir)

    result_df = pd.DataFrame()

    for station in stations:
        if os.path.isdir(station):
            station_csv = os.path.join(station, im_name + ".csv")
            im_recorder_glob = os.path.join(station, "env*/*.out")
            im_recorders = glob.glob(im_recorder_glob)
            station_name = os.path.basename(station)
            value_dict = {"station": station_name}
            for im_recorder in im_recorders:
                im_name = os.path.splitext(os.path.basename(im_recorder))[0]
                im_value = read_out_file(im_recorder)
                value_dict[im_name] = im_value
            print(value_dict)
            result_df = result_df.append(value_dict, ignore_index=True)
        cols_ex_station = list(result_df.columns.values)
        cols_ex_station.remove("station")
        result_df[result_df.station == station_name].to_csv(
            station_csv, columns=cols_ex_station, index=False
        )
    cols = ["station"]
    cols.extend(cols_ex_station)
    result_df.to_csv(im_csv_fname, index=False, columns=cols)


def read_out_file(file):
    with open(file) as f:
        lines = f.readlines()
        value = lines[-1].split()[1]
        return value


if __name__ == "__main__":
    run()
