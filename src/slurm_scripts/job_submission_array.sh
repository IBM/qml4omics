#!/bin/sh


#SBATCH -n 1 
#SBATCH -c 9
#SBATCH --mem=12000
#SBATCH -p bigmem
#SBATCH --mail-user= ## your email
#SBATCH --mail-type=BEGIN,END
#SARRAY --range=1-1728

#SBATCH --array=1-1728%5

base=$('pwd')

echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST

#### Get start time
start=`date +%s`

##### This will run in batch mode, processing batches of yamls at a time. 
srun -n 1 -c 9 python qml4omics-profiler.py --config-path=path/to/experiment/config/folder/ --config-name=exp_${SLURM_ARRAYID}.yaml

##### Get end time
end=`date +%s`
runtime=$((end-start))
echo "Time: ${runtime}"



