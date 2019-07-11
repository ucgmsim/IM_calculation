import argparse
import glob
import os
import pandas as pd
import subprocess



DEFAULT_OPEN_SEES_PATH = "OpenSees"

model_dir = os.path.dirname(__file__)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("comp_000", help="filepath to a station's 000 waveform ascii file")
    parser.add_argument("comp_090", help="filepath to a station's 090 waveform ascii file")
    parser.add_argument("comp_ver", help="filepath to a station's ver waveform ascii file")
    parser.add_argument("output_dir", help="Where the IM_csv file is written to. Also contains the temporary recorders output")

    parser.add_argument("--OpenSees_path", default=DEFAULT_OPEN_SEES_PATH, help="Path to OpenSees binary")



    args = parser.parse_args()


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)



    script = [

        args.OpenSees_path,

        os.path.join(model_dir, 'run_script.tcl'),

        args.comp_000,

        args.output_dir,

    ]



    print(' '.join(script))

    subprocess.call(script)



    create_im_csv(args.output_dir, "Steel_MF_5Story")



def create_im_csv(output_dir, im_name):
    success_glob = os.path.join(output_dir, "Analysis_*")
    success_files = glob.glob(success_glob)
    model_converged = False
    for f in success_files:
        with open(f) as fp:
            contents = fp.read()
        model_converged = model_converged or (contents.strip() == "Successful")

    im_csv_fname = os.path.join(output_dir, im_name + ".csv")
    result_df = pd.DataFrame()
    im_recorder_glob = os.path.join(output_dir, 'env*/*.out')
    im_recorders = glob.glob(im_recorder_glob)
    value_dict = {}

    for im_recorder in im_recorders:
        im_name = os.path.splitext(os.path.basename(im_recorder))[0]
        im_value = read_out_file(im_recorder, model_converged)

        value_dict[im_name] = im_value

    result_df = result_df.append(value_dict, ignore_index=True)
    # print(result_df)
    result_df.to_csv(im_csv_fname, index=False)



def read_out_file(file, success=True):
    if success:
        with open(file) as f:

            lines = f.readlines()

            value = lines[-1].split()[1]

            return value
    else:
       return float("NaN")



if __name__ == "__main__":

    main()


