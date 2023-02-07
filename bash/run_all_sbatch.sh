#!/bin/bash 
#SBATCH --job-name=isorank
#SBATCH --time=1-10:00:00
#SBATCH --mem=128G
#SBATCH --partition=preempt

ALPHA=${1}
ITER=${2}
sbatch -o out_${ALPHA}_${ITER}.log --mem 128000 --partition preempt --time=1-10:00:00 --job-name=isorank --partition=preempt ./run_tests.sh -a $ALPHA -b $ITER
