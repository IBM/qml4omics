#!/bin/sh


#SBATCH -n 1
#SBATCH -c 9
#SBATCH --mem=8000
#SBATCH -p defq
#SBATCH --mail-user= ## your email
#SBATCH --mail-type=BEGIN,END

base=$('pwd')

echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST

#### Get start time
start=`date +%s`

##### Do some calculations
srun -n 1 -c 24 python qml4omics-profiler.py --config-name=basic.yaml

##### Get end time
end=`date +%s`
runtime=$((end-start))
echo "Time: ${runtime}"



