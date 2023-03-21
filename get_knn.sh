#!/bin/bash

SPECIES=(bakers rat fly mouse)
FILES=(data/intact_output/bakers.s.tsv data/intact_output/rat.s.tsv data/intact_output/fly.s.tsv data/intact_output/mouse.s.tsv)

K=(5 10 15 100)

THRES_DSD_DIST=10

LOGPREF=$(date | tr ' ' '-' | awk '{printf "log-%s",$0}')

for i in $(seq 0 $((${#SPECIES[@]} - 1)))
do
    for k in ${K[@]}
    do 
        SP=${SPECIES[i]}
        PPI=${FILES[i]}
        DSD=mundo2data/DSD-DIST-${SP}.npy
        JSON=mundo2data/${SP}.json

        # TRANSFORMEDB_A="mundo2data/TRANSFORMED-${SPB}->${SPA}-${MDSDIM}-${NOLANDMARKS}.npy"
        # CMD="./run_mundo2.py --ppiA ${PPIA} --ppiB ${PPIB} --nameA ${SPA} --nameB ${SPB} --dsd_A_dist ${DSDA} --dsd_B_dist ${DSDB} --thres_dsd_dist ${THRES_DSD_DIST} --json_A ${JSONA} --json_B ${JSONB} --mds_A ${MDSA} --mds_B ${MDSB} --mds_r ${MDSDIM} --landmarks_a_b ${LANDMARK} --no_landmarks ${NOLANDMARKS} --model ${MODEL} --transformed_b_a ${TRANSFORMEDB_A} --mds_dist_a_b ${MDSDISTA_B} --compute_go_eval --kA ${KA} --kB ${KB} --metrics ${METRICS} --output_file outputs.tsv --go_A ${GOA} --go_B ${GOB} --compute_isorank --wB ${WEIGHTMUNDO}"
        # CMD="python get_knn.py --ppi ${PPI} --name ${SP} --dsd ${DSD} --num_neighbors $k --json ${JSON} --thres_dsd_dist ${THRES_DSD_DIST}"
        CMD="./get_knn.py --ppi ${PPI} --name ${SP} --dsd ${DSD} --num_neighbors $k --json ${JSON} --thres_dsd_dist ${THRES_DSD_DIST}"
        # Save the command that is just run
        echo ${LOGPREF} >> mundo2logs/knn_commands.txt
        echo $CMD >> mundo2logs/knn_commands.txt
        echo "Queuing command: $CMD"
        if [ -z $MODE ]
            then
                sbatch -o mundo2logs/${SP}_${k}.log --mem 128000 --partition preempt --time=1-10:00:00 --job-name=knn-${SP}-${k} --partition=preempt $CMD
            else
                $CMD
                exit
        fi
    
    # sbatch -o mundo2logs/${LOGPREF}_${SPA}_${SPB}.log --mem 128000 --partition preempt --time=1-10:00:00 --job-name=isorank-${SPA}-${SPB} --partition=preempt $CMD
    done
done