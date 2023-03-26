#!/bin/bash

SPECIESA=(bakers)
FILESA=(data/intact_output/bakers.s.tsv)

MODEL=dscript/models/topsy_turvy_v1.sav
K=15
LOGPREF=$(date | tr ' ' '-' | awk '{printf "log-%s",$0}')

for i in $(seq 0 $((${#SPECIESA[@]} - 1)))
do
    SP=${SPECIESA[j]}
    SEQS=data/seqs/${SP}_trimmed.fasta
    PAIRS=data/pairs/${SP}_${K}.tsv
    EMB=data/embed/${SP}.h5
    OUT=data/output/${SP}_${K}
    
    #CMD="/cluster/tufts/cowenlab/.envs/dscript/bin/dscript embed --seqs ${SEQS} -o ${OUT} -d 0"
    CMD="/cluster/tufts/cowenlab/.envs/dscript/bin/dscript predict --model ${MOD} --pairs ${PAIRS} --embeddings ${EMB} -o ${OUT} -d 0"
 
    # Save the command that is just run
    #echo $CMD >> mundo2logs/${LOGPREF}_${SPA}_${SPB}.cmd
    echo "Queuing command: $CMD"
       
    if [ -z $MODE ]
    then    
        sbatch -o mundo2logs/DSCRIPT_${SP}_${K}.log --mem 4g --gres=gpu:1 --partition preempt --time=1-10:00:00 --job-name=dscript-${SP}-${K} --partition=preempt $CMD
    else
        $CMD
        exit
    fi

done

