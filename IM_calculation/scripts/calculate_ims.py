# TODO when only geom is needed, only 090 and 000 should be calculated
"""
Calculate im values.
Output computed measures to /home/$user/computed_measures if no output path is specified
command:
   python calculate_ims.py test/test_calculate_ims/sample1/input/single_files/ a
   python calculate_ims.py ../BB.bin b
   python calculate_ims.py ../BB.bin b -o /home/yzh231/ -i Albury_666_999 -r Albury -t s -v 18p3 -n 112A CMZ -m PGV pSA -p 0.02 0.03 -e -c geom -np 2
"""

import argparse
import os
import pandas as pd
import glob

import IM_calculation.IM.im_calculation as calc
from IM_calculation.Advanced_IM import advanced_IM_factory
from qcore import utils
from qcore import constants


def load_args():
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
        required=True,
        help="path to output folder that stores the computed measures. Folder name must "
        "not be inclusive.eg.home/tt/. Required parameter",
    )
    parser.add_argument(
        "-i",
        "--identifier",
        required=True,
        help="Please specify the unique runname of the simulation. " "eg.Albury_REL01",
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
        "-m",
        "--im",
        nargs="+",
        default=calc.DEFAULT_IMS,
        choices=calc.ALL_IMS,
        help="Please specify im measure(s) separated by a space(if more than one). eg: PGV PGA CAV",
    )
    parser.add_argument(
        "-p",
        "--period",
        nargs="+",
        default=constants.DEFAULT_PSA_PERIODS,
        type=float,
        help="Please provide pSA period(s) separated by a space. eg: "
        "0.02 0.05 0.1. Default periods are: {} (also used for SDI).".format(
            ",".join(str(v) for v in constants.DEFAULT_PSA_PERIODS)
        ),
    )
    parser.add_argument(
        "-e",
        "--extended_period",
        action="store_true",
        help="Please add '-e' to indicate the use of extended(100) pSA periods (also used for SDI)."
        "Default not using",
    )
    parser.add_argument(
        "--fas_frequency",
        nargs="+",
        default=calc.FAS_FREQUENCY,
        type=float,
        help="Please provide fourier spectrum frequencies separated by a space. eg: "
        "0.1 0.2 0.4",
    )
    # parser.add_argument(
    #     "--extended_fas_frequency",
    #     "-f",
    #     action="store_true",
    #     help="Please add '-f' to indicate the use of extended(100) FAS frequencies. Default not using",
    # )
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
        choices=list(constants.Components.iterate_str_values()),
        help="Please provide the velocity/acc component(s) you want to calculate eg.geom."
        " Available components are: {} components. Default is geom".format(
            ",".join(constants.Components.iterate_str_values())
        ),
    )
    parser.add_argument(
        "-np",
        "--process",
        default=1,
        type=int,
        help="Please provide the number of processors. Default is 1",
    )

    parser.add_argument(
        "--real_stats_only",
        action="store_true",
        help="Please add '--real_stats_only' to consider real stations only",
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

    if args.advanced_ims is not None and args.components is not None:
        parser.error(
            "-c (--components) and -a (--advanced_ims) can not be both specified"
        )

    if args.components is None:
        args.components = [constants.Components.cgeom.str_value]

    if constants.Components.ceas.str_value in args.components:
        wrong_ims = ""
        if "FAS" not in args.im:
            wrong_ims = ",".join(args.im)
        else:
            # there are IMs other than FAS but user only wants EAS
            # without this check, the code still runs, but the non-FAS IMs
            # won't be shown in the output csv without a notice/warning
            if len(args.im) > 1 and len(args.components) == 1:
                args.im.remove("FAS")
                wrong_ims = ",".join(args.im)
        if wrong_ims:
            parser.error(
                "The specified IMs need non-EAS components to proceed: {}".format(
                    wrong_ims
                )
            )

    calc.validate_input_path(parser, args.input_path, args.file_type)

    return args


def main():
    args = load_args()

    file_type = calc.FILE_TYPE_DICT[args.file_type]
    run_type = calc.META_TYPE_DICT[args.run_type]

    im = args.im

    im_options = {}

    valid_periods = calc.validate_period(args.period, args.extended_period)
    if "pSA" in im:
        im_options["pSA"] = valid_periods
    if "SDI" in im:
        im_options["SDI"] = valid_periods
    if "FAS" in im:
        im_options["FAS"] = calc.validate_fas_frequency(args.fas_frequency)

    # Create output dir
    utils.setup_dir(args.output_path)
    if not args.simple_output:
        utils.setup_dir(os.path.join(args.output_path, calc.OUTPUT_SUBFOLDER))

    # TODO: this may need to be updated to read file if the length of list becomes an issue
    station_names = args.station_names
    if args.advanced_ims is not None:
        components = advanced_IM_factory.COMP_DICT.keys()
        advanced_im_config = advanced_IM_factory.advanced_im_config(
            args.advanced_ims, args.advanced_im_config, args.OpenSees_path
        )

    else:
        components = args.components
        advanced_im_config = None

    # multiprocessor
    calc.compute_measures_multiprocess(
        args.input_path,
        file_type,
        wave_type=None,
        station_names=station_names,
        ims=im,
        comp=components,
        im_options=im_options,
        output=args.output_path,
        identifier=args.identifier,
        rupture=args.rupture,
        run_type=run_type,
        version=args.version,
        process=args.process,
        simple_output=args.simple_output,
        units=args.units,
        advanced_im_config=advanced_im_config,
        real_only=args.real_stats_only,
    )

    print("Calculations are output to {}".format(args.output_path))


if __name__ == "__main__":
    main()
