#!/bin/bash

SPA=(bakers)
SPB=(fly)
KNNA=('' _5  _10 _15)
# KNNA=('')
# KNNB=('' _5 _10 _15)
KNNB=('')

KA="10,50"
KB="20,100"

THRES_DSD_DIST=10
METRICS="top-1-acc,f1max"
MDSDIM=100
NOLANDMARKS=500
WEIGHTMUNDO=1.5

LOGPREF=$(date | tr ' ' '-' | awk '{printf "log-%s",$0}')

for i in $(seq 0 $((${#KNNA[@]} - 1)))
do
    EXTA=${KNNA[i]}
    PPIA=data/intact_output/${SPA}${EXTA}.s.tsv
    for j in $(seq 0 $((${#KNNB[@]} - 1)))
    do
        EXTB=${KNNB[j]}
        PPIB=data/intact_output/${SPB}${EXTB}.s.tsv
        echo $PPIB
        DSDA=mundo2data/DSD-DIST-${SPA}${EXTA}.npy
        DSDB=mundo2data/DSD-DIST-${SPB}${EXTB}.npy
        JSONA=mundo2data/${SPA}.json
        JSONB=mundo2data/${SPB}.json
        GOA=data/go/${SPA}.output.mapping.gaf 
        GOB=data/go/${SPB}.output.mapping.gaf 
        MDSA=mundo2data/ISOMAP-${SPA}-${MDSDIM}${EXTA}.npy
        MDSB=mundo2data/ISOMAP-${SPB}-${MDSDIM}${EXTB}.npy
        MODEL="mundo2data/MODEL-${SPB}${EXTB}-${SPA}${EXTA}-${MDSDIM}-${NOLANDMARKS}.sav"
       
        LANDMARK=data/intact_output/${SPA}-${SPB}.tsv
        if [ ! -f $LANDMARK ]; then LANDMARK=data/intact_output/${SPB}-${SPA}.tsv; fi
       
        TRANSFORMEDB_A="mundo2data/TRANSFORMED-${SPB}${EXTB}-${SPA}${EXTA}-${MDSDIM}-${NOLANDMARKS}.npy"
        MDSDISTA_B="mundo2data/MDS-DIST-${SPB}${EXTB}-${SPA}${EXTA}-${MDSDIM}-${NOLANDMARKS}.npy"
        CMD="./run_mundo2.py --ppiA ${PPIA} --ppiB ${PPIB} --nameA ${SPA} --nameB ${SPB} --dsd_A_dist ${DSDA} --dsd_B_dist ${DSDB} --thres_dsd_dist ${THRES_DSD_DIST} --json_A ${JSONA} --json_B ${JSONB} --mds_A ${MDSA} --mds_B ${MDSB} --mds_r ${MDSDIM} --landmarks_a_b ${LANDMARK} --no_landmarks ${NOLANDMARKS} --model ${MODEL} --transformed_b_a ${TRANSFORMEDB_A} --mds_dist_a_b ${MDSDISTA_B} --compute_go_eval --kA ${KA} --kB ${KB} --metrics ${METRICS} --output_file outputs.tsv --go_A ${GOA} --go_B ${GOB} --compute_isorank --wB ${WEIGHTMUNDO}"
       
    #    # Save the command that is just run
    #    # echo $CMD >> mundo2logs/base_${SPA}_${SPB}.cmd
        echo "Queuing command: $CMD"
       
        sbatch -o mundo2logs/addition_${SPA}${EXTA}_${SPB}${EXTB}.log --mem 128000 --partition preempt --time=1-10:00:00 --job-name=isorank-${SPA}-${SPB} --partition=preempt $CMD
    done
done
