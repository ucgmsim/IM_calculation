import pandas as pd
import argparse
from pathlib import Path

g=9.81

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Script for retrieving IESD outputs from the single IM csv and storing them in separate files depending on model"
    )
    parser.add_argument(
        "csv_file", type=str, help=".csv"
    )

    parser.add_argument(
        "-o", "--outdir", type=str, help="Directory for the output file"
    )
    args = parser.parse_args()

    args.csv_file = Path(args.csv_file).resolve()

    filename = args.csv_file.name

    if args.outdir is None:
        args.outdir = args.csv_file.parent


    print(args.outdir)

    df = pd.read_csv(args.csv_file,index_col=[0,1])
    df.describe()
    colnames = list(df.columns)

    categories = {}
    for colname in colnames:
        chunks = colname.split('_')
        model='_'.join(chunks[:-1])
        period = chunks[-1]
        if model not in categories:
            categories[model]=[period]
        else:
            categories[model].append(period)

    for model,periods in categories.items():
        new_df = df[["{}_{}".format(model,period) for period in periods]].copy(deep=True)
        new_df.rename(columns=dict(zip(new_df.columns,periods)),inplace=True)
        new_df = new_df * g
        new_df.to_csv(args.outdir / Path(model+".csv"))