#!/bin/bash

#SBATCH --account=hai_dnn_gll
#SBATCH --nodes=1
#SBATCH --output=outputs/slurm_outputs/slurm_%j.out
#SBATCH --error=outputs/slurm_outputs/slurm_%j.out
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:4
#SBATCH --job-name=haicore_tune_dp
#SBATCH --partition=booster
#SBATCH --mail-type=ALL

CWD="$PROJECT/automatic_gll_delineation/ml"
echo "CWD = $CWD"
cd $CWD

#Activate python module
module load Python

#Activate virutal environment
source $PROJECT/automatic_gll_delineation/.venv/bin/activate


srun --exclusive -n 1 --gres=gpu:1 -o $PROJECT/outputs/snakemake_stdout/%j_1.out snakemake --nolock -R predict --configfile ../configs/config_1.yaml --cores 64 --quiet &
sleep 60s
srun --exclusive -n 1 --gres=gpu:1 -o $PROJECT/outputs/snakemake_stdout/%j_2.out snakemake --nolock -R predict --configfile ../configs/config_2.yaml --cores 64 --quiet &
sleep 60s
srun --exclusive -n 1 --gres=gpu:1 -o $PROJECT/outputs/snakemake_stdout/%j_3.out snakemake --nolock -R predict --configfile ../configs/config_3.yaml --cores 64 --quiet &
sleep 60s
srun --exclusive -n 1 --gres=gpu:1 -o $PROJECT/outputs/snakemake_stdout/%j_4.out snakemake --nolock -R predict --configfile ../configs/config_4.yaml --cores 64 --quiet &

wait
