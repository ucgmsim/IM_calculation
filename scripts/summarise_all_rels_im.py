import pandas as pd
import numpy as np
import glob
import os
import sys
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Compute logarithmic mean/median of IM values across realisations of a given fault")
    parser.add_argument('csv_path',help='path to CSV files to recursively search') #if set too deep, search will be slow.
    parser.add_argument('fault_name',help='name of the fault to process')
    parser.add_argument('--median', dest='stat_fn',action='store_const', const= np.median, default=np.mean, help='find median (default: mean)') 
    parser.add_argument('--im',dest='im_types', nargs='+', type=str, help='list of IM types. all chosen if unspecified') #eg. --im PGV PGA
    parser.add_argument('--output', dest='output_dir', help='path for the output CSV file to be saved')

    args = parser.parse_args()

    csv_path = args.csv_path
    
    fault_name = args.fault_name
    stat_fn = args.stat_fn #default: np.mean
    im_types = args.im_types
    output_dir = args.output_dir

    csv_path = os.path.abspath(csv_path)
    if os.path.exists(csv_path) and os.path.isdir(csv_path):
        print("Checked: CSV search directory {}".format(csv_path))
    else:
        print("Error: invalid path : {}".format(csv_path))
        sys.exit(0)

    
    im_csv_paths = glob.glob("{}/**/{}_REL*.csv".format(csv_path,fault_name),recursive=True)
    im_csv_paths.sort()

    if len(im_csv_paths) > 0:
        print("Checked: IM csv files located")
    else:
        print("Error: no IM csv files found")
        sys.exit(0)


    print(im_csv_paths)

    if output_dir is None:
        output_dir = os.path.curdir
    output_dir = os.path.abspath(output_dir)
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        print("Checked: Output directory {}".format(output_dir))
    else:
        os.mkdir(output_dir)
        print("Created: Output directory {}".format(output_dir))
    
    rel_im_dfs=[]
    for c in im_csv_paths:
        df = pd.read_csv(c,index_col=0)
        rel_im_dfs.append(df)

    stations=list(rel_im_dfs[0].index)
    stations.sort()

    # check IM types. If unspecified, use all IM_types
    wrong_im_count=0
    if im_types is None:
        im_types = list(rel_im_dfs[0].columns)
        im_types.remove("component")
    else:
        for im_type in im_types:
            if im_type in rel_im_dfs[0].columns:
                pass
            else:
                print("Error: Unknown IM type {}".format(im_type))
                wrong_im_count+=1
        if wrong_im_count > 0:
            print("Error: Fix IM types")
            sys.exit(0)

    print("Summarising IM values at {} stations from {} realisations for IM types {}".format(len(stations),len(rel_im_dfs), im_types))
    df_dict = {'station':stations,'component':['geom']*len(stations)}
    for im_type in im_types:
        print("...{}".format(im_type, len(stations), len(rel_im_dfs)))
        im_val_concat = pd.concat([np.log(rel_im_dfs[i][im_type]) for i in range(len(rel_im_dfs))])

        log_mean_im=[np.exp(stat_fn(im_val_concat[k])) for k in stations]
        df_dict[im_type]=log_mean_im

    log_mean_ims_df=pd.DataFrame(df_dict)
    output_file = os.path.join(output_dir,fault_name+'_log_{}.csv'.format(stat_fn.__name__))

    log_mean_ims_df.set_index('station').to_csv(output_file)
    print("Completed...Written {}".format(output_file))
    
