#!/bin/sh


#SBATCH -n 1
#SBATCH -c 9
#SBATCH --mem=4000
#SBATCH -p defq
#SBATCH --mail-user= ## your email
#SBATCH --mail-type=BEGIN,END

base=$('pwd')

echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST

#### Get start time
start=`date +%s`

##### Loop over input yamls -- this may be happening serially
for i in {1..864} # this range should be the same as the number of input yaml files in config folder
do
   python qml4omics-profiler.py --config-path=path/to/experiment/config/folder/ --config-name=exp_${i}.yaml
done

##### Get end time
end=`date +%s`
runtime=$((end-start))
echo "Time: ${runtime}"



