#!/bin/bash

#SBATCH --account=engin1
#SBATCH --job-name=main
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=8GB
#SBATCH --time=00-1:00:00
#SBATCH --mail-user=rivachen@umich.edu
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --partition=standard
#SBATCH --array=1-10
#SBATCH --output="/home/rivachen/GP_MNIST/FirstDerivative_10labels"

#
# Include the next three lines always
if [ "x${PBS_NODEFILE}" != "x" ]; then
    cat $PBS_NODEFILE
fi

cd /home/rivachen/GP_MNIST/FirstDerivative_10labels

module load python
python test.py ./${SLURM_ARRAY_TASK_ID}.out

