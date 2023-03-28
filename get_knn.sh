#!/bin/bash

SPECIES=(bakers rat fly)
FILES=(data/intact_output/rat.s.tsv)

K=("" _5 _10 _15)


THRES_DSD_DIST=10

LOGPREF=$(date | tr ' ' '-' | awk '{printf "log-%s",$0}')

for i in $(seq 0 $((${#SPECIES[@]} - 1)))
do
    SP=${SPECIES[i]}
    for j in $(seq 0 $((${#K[@]} - 1)))
    do 
        k=${K[j]}
        PPI=data/intact_output/${SP}${k}.s.tsv
        DSD=mundo2data/DSD-DIST-${SP}${k}.npy
        JSON=mundo2data/${SP}.json

        CMD="./get_knn.py --ppi ${PPI} --name ${SP} --dsd ${DSD} --ext $k --json ${JSON} --thres_dsd_dist ${THRES_DSD_DIST}"
        # echo "Queuing command: $CMD"
        if [ -z $MODE ]
            then
                sbatch -o mundo2logs/${SP}_${k}.log --mem 128000 --partition preempt --time=1-10:00:00 --job-name=knn-${SP}-${k} --partition=preempt $CMD
            else
                $CMD
                exit
        fi
    done
done
