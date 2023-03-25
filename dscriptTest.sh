#!/bin/bash #required
#SBATCH -N 1 # nodes requested
#SBATCH -n 1 # tasks requested
#SBATCH -p gpu # tasks requested
#SBATCH --mem=50g # memory in Mb
#SBATCH -o data/output/test_run

module load anaconda/2021.11
module load cuda/10.0
conda activate /cluster/tufts/cowenlab/.envs/dscript/

K=15
SP=fly
PAIRS=data/pairs/${SP}_{K}.tsv
SEQS=data/pairs/${SP}_${k}.tsv
EMB=data/embed/${SP}.tsv
OUT=data/output/${SP}_${K}
MODEL=dscript/models/topsy_turvy_v1.sav


python -m dscript predict --pairs ${PAIRS} --seqs ${SEQS} -o ${OUT} --model ${MODEL} -d 0