import csv
import getpass
import glob
import os
import sys
from datetime import datetime
import numpy as np

from IM_calculation.IM import read_waveform, intensity_measures
from qcore import timeseries, pool_wrapper
from IM_calculation.Advanced_IM import advanced_IM_factory

G = 981.0
IMS = ["PGA", "PGV", "CAV", "AI", "Ds575", "Ds595", "MMI", "pSA"]

EXT_PERIOD = np.logspace(start=np.log10(0.01), stop=np.log10(10.0), num=100, base=10)
BSC_PERIOD = [
    0.02,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.75,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    7.5,
    10.0,
]

COMPONENTS = ["090", "000", "ver", "geom"]

FILE_TYPE_DICT = {"a": "ascii", "b": "binary"}
META_TYPE_DICT = {"s": "simulated", "o": "observed", "u": "unknown"}
RUNNAME_DEFAULT = "all_station_ims"

OUTPUT_PATH = os.path.join("/home", getpass.getuser())
OUTPUT_SUBFOLDER = "stations"

MEM_PER_CORE = 7.5e8
MEM_FACTOR = 4


def convert_str_comp(arg_comps):
    """
    convert arg comps to str_comps_for integer_convestion in read_waveform & str comps for writing result
    :param arg_comps: user input a list of comp(s)
    :return: two lists of str comps
    """
    # comps = ["000", "geom",'ver']
    if "geom" in arg_comps:
        # ['000', 'ver', '090', 'geom']
        str_comps = list(set(arg_comps).union({"090", "000"}))
        # for integer convention, str_comps shouldn't include geom as waveform has max 3 components
        str_comps.remove("geom")
        # for writing result, make a copy of the str_comps for int convention, and shift geom to the end
        str_comps_for_writing = str_comps[:]
        str_comps_for_writing.append("geom")
        return str_comps, str_comps_for_writing
    else:
        return arg_comps, arg_comps


def array_to_dict(value, str_comps, im, arg_comps):
    """
    convert a numpy arrary that contains calculated im values to a dict {comp: value}
    :param value: calculated intensity measure for a waveform
    :param str_comps: a list of components converted from user input
    :param im:
    :param arg_comps:user input list of components
    :return: a dict {comp: value}
    """
    value_dict = {}
    # ["090", "ver"], ["090", "000", "geom"], ["090", "000", "ver", "geom"]
    # [0, 2]          [0, 1]                  [0, 1, 2]
    for i in range(value.shape[-1]):
        # pSA returns a 2D array
        if im == "pSA":
            value_dict[str_comps[i]] = value[:, i]
        else:
            value_dict[str_comps[i]] = value[i]
    # In this case, if geom in str_comps,
    # it's guaranteed that 090 and 000 will be present in value_dict
    if "geom" in str_comps:
        value_dict["geom"] = intensity_measures.get_geom(
            value_dict["090"], value_dict["000"]
        )
        # then we pop unwanted keys from value_dict
        for k in str_comps:
            if k not in arg_comps:
                del value_dict[k]
    return value_dict


def compute_measure_single(value_tuple):
    """
    Compute measures for a single station
    :param: a tuple consisting 6 params: waveform, ims, comp, period, str_comps, advanced_ims
    waveform: a single tuple that contains (waveform_acc,waveform_vel)
    :return: {result[station_name]: {[im]: value or (period,value}}
    """
    waveform, ims, comps, period, str_comps, advanced_im_config = value_tuple
    result = {}
    waveform_acc, waveform_vel = waveform
    DT = waveform_acc.DT
    times = waveform_acc.times

    accelerations = waveform_acc.values

    if waveform_vel is None:
        # integrating g to cm/s
        velocities = timeseries.acc2vel(accelerations, DT) * G
    else:
        velocities = waveform_vel.values

    station_name = waveform_acc.station_name

    result[station_name] = {}

    for im in ims:
        if im == "PGV":
            value = intensity_measures.get_max_nd(velocities)

        if im == "PGA":
            value = intensity_measures.get_max_nd(accelerations)

        if im == "pSA":
            value = intensity_measures.get_spectral_acceleration_nd(
                accelerations, period, waveform_acc.NT, DT
            )

        # TODO: Speed up Ds calculations
        if im == "Ds595":
            value = intensity_measures.getDs_nd(DT, accelerations, 5, 95)

        if im == "Ds575":
            value = intensity_measures.getDs_nd(DT, accelerations, 5, 75)

        if im == "AI":
            value = intensity_measures.get_arias_intensity_nd(accelerations, G, times)

        if im == "CAV":
            value = intensity_measures.get_cumulative_abs_velocity_nd(
                accelerations, times
            )

        if im == "MMI":
            value = intensity_measures.calculate_MMI_nd(velocities)

        # store a im type values into a dict {comp: np_array/single float}
        # Geometric is also calculated here
        value_dict = array_to_dict(value, str_comps, im, comps)

        # store value dict into the biggest result dict
        if im == "pSA":
            result[station_name][im] = (period, value_dict)
        else:
            result[station_name][im] = value_dict

    if advanced_im_config is not None:
        advanced_IM_factory.compute_ims(waveform_acc, advanced_im_config)

    return result


def get_bbseis(input_path, file_type, selected_stations):
    """
    :param input_path: user input path to bb.bin or a folder containing ascii files
    :param file_type: binary or ascii
    :param selected_stations: list of user input stations
    :return: bbseries, station_names
    """
    bbseries = None
    if file_type == FILE_TYPE_DICT["b"]:
        bbseries = timeseries.BBSeis(input_path)
        if selected_stations is None:
            station_names = bbseries.stations.name
        else:
            station_names = selected_stations
    elif file_type == FILE_TYPE_DICT["a"]:
        search_path = os.path.abspath(os.path.join(input_path, "*"))
        files = glob.glob(search_path)
        station_names = set(map(read_waveform.get_station_name_from_filepath, files))
        if selected_stations is not None:
            station_names = station_names.intersection(selected_stations)
            if len(station_names) == 0:  # empty set
                sys.exit(
                    "could not find specified stations {} in folder {}".format(
                        selected_stations, input_path
                    )
                )
    return bbseries, list(station_names)


def compute_measures_multiprocess(
    input_path,
    file_type,
    wave_type,
    station_names,
    ims=IMS,
    comp=None,
    period=None,
    output=None,
    identifier=None,
    rupture=None,
    run_type=None,
    version=None,
    process=1,
    simple_output=False,
    units="g",
    advanced_im_config=None,
):
    """
    using multiprocesses to compute measures.
    Calls compute_measure_single() to compute measures for a single station
    write results to csvs and an imcalc.info meta data file
    :param input_path:
    :param file_type:
    :param wave_type:
    :param station_names:
    :param ims:
    :param comp:
    :param period:
    :param output:
    :param identifier:
    :param rupture:
    :param run_type:
    :param version:
    :param process:
    :param simple_output:
    :return:
    """
    str_comps_for_int, str_comps = convert_str_comp(comp)

    bbseries, station_names = get_bbseis(input_path, file_type, station_names)

    total_stations = len(station_names)
    steps = get_steps(input_path, process, total_stations)

    all_result_dict = {}
    p = pool_wrapper.PoolWrapper(process)

    i = 0
    while i < total_stations:
        waveforms = read_waveform.read_waveforms(
            input_path,
            bbseries,
            station_names[i : i + steps],
            str_comps_for_int,
            wave_type=wave_type,
            file_type=file_type,
            units=units,
        )
        i += steps
        array_params = []
        for waveform in waveforms:
            array_params.append(
                (waveform, ims, comp, period, str_comps, advanced_im_config)
            )
        result_list = p.map(compute_measure_single, array_params)

        for result in result_list:
            all_result_dict.update(result)

    write_result(all_result_dict, output, identifier, comp, ims, period, simple_output)

    generate_metadata(output, identifier, rupture, run_type, version)


def get_result_filepath(output_folder, arg_identifier, suffix):
    return os.path.join(output_folder, "{}{}".format(arg_identifier, suffix))


def get_header(ims, period):
    """
    write header colums for output im_calculations csv file
    :param ims: a list of im measures
    :param period: a list of pSA periods
    :return: header
    """
    header = ["station", "component"]
    psa_names = []

    for im in ims:
        if im == "pSA":  # only write period if im is pSA.
            for p in period:
                if p in BSC_PERIOD:
                    psa_names.append("pSA_{}".format(p))
                else:
                    psa_names.append("pSA_{:.12f}".format(p))
            header += psa_names
        else:
            header.append(im)
    return header


def write_rows(comps, station, ims, result_dict, big_csv_writer, sub_csv_writer=None):
    """
    write rows to big csv and, also to single station csvs if not simple output
    :param comps:
    :param station:
    :param ims:
    :param result_dict:
    :param big_csv_writer:
    :param sub_csv_writer:
    :return: write a single row
    """
    for c in comps:
        row = [station, c]
        for im in ims:
            if im != "pSA":
                row.append(result_dict[station][im][c])
            else:
                row += result_dict[station][im][1][c].tolist()
        big_csv_writer.writerow(row)
        if sub_csv_writer:
            sub_csv_writer.writerow(row)


def write_result(
    result_dict, output_folder, identifier, comps, ims, period, simple_output
):
    """
    write a big csv that contains all calculated im value and single station csvs
    :param result_dict:
    :param output_folder:
    :param identifier: user input run name
    :param comps: a list of comp(s)
    :param ims: a list of im(s)
    :param period:
    :param simple_output
    :return:output result into csvs
    """
    output_path = get_result_filepath(output_folder, identifier, ".csv")
    header = get_header(ims, period)

    # big csv containing all stations
    with open(output_path, "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",", quotechar="|")
        csv_writer.writerow(header)
        stations = sorted(result_dict.keys())

        # write each row
        for station in stations:
            if not simple_output:  # if single station csvs are needed
                station_csv = os.path.join(
                    output_folder,
                    OUTPUT_SUBFOLDER,
                    "{}_{}.csv".format(station, "_".join(comps)),
                )
                with open(station_csv, "w") as sub_csv_file:
                    sub_csv_writer = csv.writer(
                        sub_csv_file, delimiter=",", quotechar="|"
                    )
                    sub_csv_writer.writerow(header)
                    write_rows(
                        comps,
                        station,
                        ims,
                        result_dict,
                        csv_writer,
                        sub_csv_writer=sub_csv_writer,
                    )
            else:  # if only the big summary csv is needed
                write_rows(comps, station, ims, result_dict, csv_writer)


def generate_metadata(output_folder, identifier, rupture, run_type, version):
    """
    write meta data file
    :param output_folder:
    :param identifier: user input
    :param rupture: user input
    :param run_type: user input
    :param version: user input
    :return:
    """
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = get_result_filepath(output_folder, identifier, "_imcalc.info")

    with open(output_path, "w") as meta_file:
        meta_writer = csv.writer(meta_file, delimiter=",", quotechar="|")
        meta_writer.writerow(["identifier", "rupture", "type", "date", "version"])
        meta_writer.writerow([identifier, rupture, run_type, date, version])


def get_im_or_period_help(default_values, im_or_period):
    """
    :param default_values: predefined constants
    :param im_or_period: should be either string "im" or string "period"
    :return: a help message for input component arg
    """
    return "Available and default {}s are: {}".format(
        im_or_period, ",".join(str(v) for v in default_values)
    )


def validate_input_path(parser, arg_input, arg_file_type):
    """
    validate input path
    :param parser:
    :param arg_input:
    :param arg_file_type:
    :return:
    """
    if not os.path.exists(arg_input):
        parser.error("{} does not exist".format(arg_input))

    if arg_file_type == "b":
        if os.path.isdir(arg_input):
            parser.error(
                "The path should point to a binary file but not a directory. "
                "Correct Sample: /home/tt/BB.bin"
            )
    elif arg_file_type == "a":
        if os.path.isfile(arg_input):
            parser.error(
                "The path should be a directory but not a file. Correct "
                "Sample: /home/tt/sims/"
            )


def validate_im(parser, arg_im):
    """
    returns validated user input if pass the validation else raise parser error
    :param parser:
    :param arg_im: input
    :return: validated im(s) in a list
    """
    im = arg_im
    if im != IMS:
        for m in im:
            if m not in IMS:
                parser.error(
                    "please enter valid im measure name. {}".format(
                        get_im_or_period_help(IMS, "IM")
                    )
                )
    return im


def validate_period(parser, arg_period, arg_extended_period, im):
    """
    returns validated user input if pass the validation else raise parser error
    :param parser:
    :param arg_period: input
    :param arg_extended_period: input
    :param im: validated im(s) in a list
    :return: period(s) in a numpy array
    """
    period = np.array(arg_period, dtype="float64")

    if arg_extended_period:
        period = np.unique(np.append(period, EXT_PERIOD))

    if (arg_extended_period or period.any()) and "pSA" not in im:
        parser.error(
            "period or extended period must be used with pSA, but pSA is not in the "
            "IM measures entered"
        )

    return period


def get_steps(input_path, nps, total_stations):
    """
    :param input_path: user input file/dir path
    :param nps: number of processes
    :param total_stations: total number of stations
    :return: number of stations per iteration/batch
    """
    estimated_mem = os.stat(input_path).st_size * MEM_FACTOR
    available_mem = nps * MEM_PER_CORE
    batches = np.ceil(np.divide(estimated_mem, available_mem))
    steps = int(np.floor(np.divide(total_stations, batches)))
    if steps == 0:
        steps = total_stations
    return steps


def main():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--advanced_im_config",
        default=advanced_IM_factory.CONFIG_FILE_NAME,
        help="Path to the advanced IM_config file",
    )

    parent_args = parent_parser.parse_known_args()

    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True)

    parser.add_argument(
        "input_path", help="path to input bb binary file eg./home/melody/BB.bin"
    )
    parser.add_argument(
        "file_type",
        choices=["a", "b"],
        help="Please type 'a'(ascii) or 'b'(binary) to indicate the type of input file",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        default=OUTPUT_PATH,
        help="path to output folder that stores the computed measures.Folder name must "
        "not be inclusive.eg.home/tt/. Default to /home/$user/",
    )
    parser.add_argument(
        "-i",
        "--identifier",
        default=RUNNAME_DEFAULT,
        help="Please specify the unique runname of the simulation. "
        "eg.Albury_HYP01-01_S1244",
    )
    parser.add_argument(
        "-r",
        "--rupture",
        default="unknown",
        help="Please specify the rupture name of the simulation. eg.Albury",
    )
    parser.add_argument(
        "-t",
        "--run_type",
        choices=["s", "o", "u"],
        default="u",
        help="Please specify the type of the simrun. Type 's'(simulated) or "
        "'o'(observed) or 'u'(unknown)",
    )
    parser.add_argument(
        "-v",
        "--version",
        default="XXpY",
        help="Please specify the version of the simulation. eg.18p4",
    )
    parser.add_argument(
        "-im",
        "--im",
        nargs="+",
        choices=IMS,
        default=IMS,
        help="Please specify im measure(s) separated by a space(if more than one). "
        "eg: PGV PGA CAV. {}".format(get_im_or_period_help(IMS, "IM")),
    )
    parser.add_argument(
        "-p",
        "--period",
        nargs="+",
        default=BSC_PERIOD,
        type=float,
        help="Please provide pSA period(s) separated by a space. eg: "
        "0.02 0.05 0.1. {}".format(get_im_or_period_help(BSC_PERIOD, "period")),
    )
    parser.add_argument(
        "-e",
        "--extended_period",
        action="store_true",
        help="Please add '-e' to indicate the use of extended(100) pSA periods. "
        "Default not using",
    )
    parser.add_argument(
        "-n",
        "--station_names",
        nargs="+",
        help="Please provide a station name(s) separated by a space. eg: 112A 113A",
    )
    parser.add_argument(
        "-c",
        "--components",
        nargs="+",
        choices=COMPONENTS,
        default=COMPONENTS,
        help="Please provide the velocity/acc component(s) you want to calculate eg.geom."
        " Available compoents are: {} components. Default is all components".format(
            ",".join(COMPONENTS)
        ),
    )
    parser.add_argument(
        "-np",
        "--process",
        default=2,
        type=int,
        help="Please provide the number of processors. Default is 2",
    )
    parser.add_argument(
        "-s",
        "--simple_output",
        action="store_true",
        help="Please add '-s' to indicate if you want to output the big summary csv "
        "only(no single station csvs). Default outputting both single station "
        "and the big summary csvs",
    )
    parser.add_argument(
        "-u",
        "--units",
        choices=["cm/s^2", "g"],
        default="g",
        help="The units that input acceleration files are in",
    )
    parser.add_argument(
        "-a",
        "--advanced_ims",
        nargs="+",
        choices=advanced_IM_factory.get_im_list(parent_args[0].advanced_im_config),
        help="Provides the list of Advanced IMs to be calculated",
    )
    parser.add_argument(
        "--OpenSees_path", default="OpenSees", help="Path to OpenSees binary"
    )

    args = parser.parse_args()

    validate_input_path(parser, args.input_path, args.file_type)
    file_type = FILE_TYPE_DICT[args.file_type]
    run_type = META_TYPE_DICT[args.run_type]
    im = validate_im(parser, args.im)
    period = validate_period(parser, args.period, args.extended_period, im)

    advanced_im_config = advanced_IM_factory.advanced_im_config(
        args.advanced_ims, args.advanced_im_config, args.OpenSees_path
    )

    # Create output dir
    utils.setup_dir(args.output_path)
    if not args.simple_output:
        utils.setup_dir(os.path.join(args.output_path, OUTPUT_SUBFOLDER))

    # multiprocessor
    compute_measures_multiprocess(
        args.input_path,
        file_type,
        wave_type=None,
        station_names=args.station_names,
        ims=im,
        comp=args.components,
        period=period,
        output=args.output_path,
        identifier=args.identifier,
        rupture=args.rupture,
        run_type=run_type,
        version=args.version,
        process=args.process,
        simple_output=args.simple_output,
        units=args.units,
        advanced_im_config=advanced_im_config,
    )

    print("Calculations are outputted to {}".format(args.output_path))


if __name__ == "__main__":
    main()
