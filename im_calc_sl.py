#!/bin/bash
#script version: slurm

# Please modify this file as needed, this is just a sample
#SBATCH --job-name=im_calc
#SBATCH --account=nesi00213
#SBATCH --partition=nesi_research
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output im_calc-%j.out
#SBATCH --error im_calc-%j.err
###SBATCH --mail-type=all
###SBATCH --mail-user=test@test.com
###SBATCH --mem-per-cpu=90G
###SBATCH -C avx
#SBATCH --hint=nomultithread

## END HEADER
date

source machine_env.sh
export IMPATH=$gmsim/IM_calculation
export PYTHONPATH=$gmsim/qcore:/$PYTHONPATH:$IMPATH

echo ___calculating simulations___
time python calculate_ims.py runs/Runs/Pahaua/BB/Cant1D_v3-midQ_OneRay_hfnp2mm+_rvf0p8_sd50_k0p045/Pahaua_HYP01-48_S1244/Acc/BB.bin b -o ~/pahaua -i pahaua -r pahaua1244 -t s -e -s -np 40



date
