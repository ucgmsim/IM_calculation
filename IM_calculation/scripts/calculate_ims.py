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
        "-im",
        "--im",
        nargs="+",
        choices=calc.IMS,
        default=calc.IMS,
        help="Please specify im measure(s) separated by a space(if more than one). "
        "eg: PGV PGA CAV. {}".format(calc.get_im_or_period_help(calc.IMS, "IM")),
    )
    parser.add_argument(
        "-p",
        "--period",
        nargs="+",
        default=calc.BSC_PERIOD,
        type=float,
        help="Please provide pSA period(s) separated by a space. eg: "
        "0.02 0.05 0.1. {}".format(
            calc.get_im_or_period_help(calc.BSC_PERIOD, "period")
        ),
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
        choices=calc.COMPONENTS,
        default=calc.COMPONENTS,
        help="Please provide the velocity/acc component(s) you want to calculate eg.geom."
        " Available compoents are: {} components. Default is all components".format(
            ",".join(calc.COMPONENTS)
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
    parser.add_argument(
        "--observed",
        default=None,
        help="path to the observed data for matching station list",
    )
    args = parser.parse_args()

    calc.validate_input_path(parser, args.input_path, args.file_type)
    file_type = calc.FILE_TYPE_DICT[args.file_type]
    run_type = calc.META_TYPE_DICT[args.run_type]
    im = calc.validate_im(parser, args.im)
    period = calc.validate_period(parser, args.period, args.extended_period, im)

    advanced_im_config = advanced_IM_factory.advanced_im_config(
        args.advanced_ims, args.advanced_im_config, args.OpenSees_path
    )

    # Create output dir
    utils.setup_dir(args.output_path)
    if not args.simple_output:
        utils.setup_dir(os.path.join(args.output_path, calc.OUTPUT_SUBFOLDER))

    if args.observed != None:
        # retreived station list from observed/fault(eventname)/Vol*/data/accBB/station.*
        # glob for potential directory
        obs_accBB_dir_glob = os.path.join(args.observed, "*/*/accBB")
        obs_accBB_dirs = glob.glob(obs_accBB_dir_glob)
        station_names = []
        # for cases that there are more than one Vol records
        for obs_accBB_dir in obs_accBB_dirs:
            _, station_names_tmp = calc.get_bbseis(
                obs_accBB_dir, calc.FILE_TYPE_DICT["a"], None
            )
            station_names = list(set(station_names + station_names_tmp))
    else:
        station_names = args.station_names
    # multiprocessor

    calc.compute_measures_multiprocess(
        args.input_path,
        file_type,
        wave_type=None,
        station_names=station_names,
        ims=im,
        comp=args.components,
        period=period,
        output_dir=args.output_path,
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
