#!/bin/bash

SPECIESA=(bakers)
FILESA=(data/intact_output/bakers.s.tsv)

K=15
LOGPREF=$(date | tr ' ' '-' | awk '{printf "log-%s",$0}')

for i in $(seq 0 $((${#SPECIESB[@]} - 1)))
do
    SP=${SPECIESA[j]}
    SEQS=data/pairs/${SP}_${k}.tsv
    OUT=data/embed/${SP}.tsv


    CMD="./run_mundo_munk.py --ppiA ${PPIA} --ppiB ${PPIB} --nameA ${SPA} --nameB ${SPB} --dsd_A_dist ${DSDA} --dsd_B_dist ${DSDB} --thres_dsd_dist ${THRES_DSD_DIST} --json_A ${JSONA} --json_B ${JSONB} --landmarks_a_b ${LANDMARK} --no_landmarks ${NOLANDMARKS} --munk_matrix $MUNK --compute_go_eval --kA ${KA} --kB ${KB} --metrics ${METRICS} --output_file ${OPFILE} --go_A ${GOA} --go_B ${GOB} --compute_isorank --wB ${WEIGHTMUNDO}"
    
    CMD="/cluster/tufts/cowenlab/.envs/dscript/bin/dscript --seqs ${SEQS} -o ${OUT} -d 0"

    # Save the command that is just run
    #echo $CMD >> mundo2logs/${LOGPREF}_${SPA}_${SPB}.cmd
    echo "Queuing command: $CMD"
       
    if [ -z $MODE ]
    then
        sbatch -o mundo2logs/MUNK_baseline_${SPA}_${SPB}.log --mem 128000 --partition preempt --time=1-10:00:00 --job-name=isorank-${SPA}-${SPB} --partition=preempt $CMD
    else
        $CMD
        exit
    fi

done

