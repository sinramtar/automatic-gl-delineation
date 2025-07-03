#!/bin/bash -l

#SBATCH --account=hpda-c
#SBATCH -D <work directory>
#SBATCH --nodes=1
#SBATCH --output=outputs/slurm_outputs/slurm_%j.out
#SBATCH --error=outputs/slurm_outputs/slurm_%j.out
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=tune
#SBATCH --clusters=hpda2
#SBATCH --partition=hpda2_compute_gpu
#SBATCH --mail-user=Sindhu.RamanathTarekere@dlr.de


CWD = <work directory>
echo "CWD = $CWD"
cd $CWD

#Activate python module
module load slurm_setup
module load miniconda3

#Activate virutal environment

srun --exclusive -n 1 --gres=gpu:1 -o ../../outputs/snakemake_outputs/%j_1.out snakemake --nolock -R train --configfile ../configs/config_1.yaml --cores 2 --quiet &
sleep 60s
wait