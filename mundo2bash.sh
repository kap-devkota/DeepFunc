#!/bin/bash

SPECIESB=(human bakers)
FILESB=(data/intact_output/human_sub.s.tsv data/intact_output/bakers.s.tsv)


SPECIESA=(fly rat mouse)
FILESA=(data/intact_output/fly.s.tsv data/intact_output/rat.s.tsv data/intact_output/mouse.s.tsv)

KA="10,20,30,40,50"
KB="10,20,30,40,50,100"

THRES_DSD_DIST=10
METRICS="top-1-acc,top-5-acc"


MDSDIM=100
NOLANDMARKS=500
WEIGHTMUNDO=0.4

while getopts "m:l:w:" args
do
    case $args in 
        m) MDSDIM=${OPTARG}
        ;;
        l) NOLANDMARKS=${OPTARG}
        ;;
	w) WEIGHTMUNDO=${OPTARG}
	;;
    esac
done

# (100, 500) RUNNING
# (150, 500) RUNNING
# (50, 500) RUNNING
# (200, 500) RUNNING
# (500, 500) RUNNING


LOGPREF=$(date | tr ' ' '-' | awk '{printf "log-%s",$0}')


for i in $(seq 0 $((${#SPECIESB[@]} - 1)))
do
    for j in $(seq 0 $((${#SPECIESA[@]} - 1)))
    do
       SPA=${SPECIESA[j]}
       SPB=${SPECIESB[i]}
       PPIA=${FILESA[j]}
       PPIB=${FILESB[i]}
       DSDA=mundo2data/DSD-DIST-${SPA}.npy
       DSDB=mundo2data/DSD-DIST-${SPB}.npy
       JSONA=mundo2data/${SPA}.json
       JSONB=mundo2data/${SPB}.json
       GOA=data/go/${SPA}.output.mapping.gaf 
       GOB=data/go/${SPB}.output.mapping.gaf 
       MDSA=mundo2data/ISOMAP-${SPA}-${MDSDIM}.npy
       MDSB=mundo2data/ISOMAP-${SPB}-${MDSDIM}.npy
       MODEL="mundo2data/MODEL-${SPB}->${SPA}-${MDSDIM}-${NOLANDMARKS}.sav"
       
       LANDMARK=data/intact_output/${SPA}-${SPB}.tsv
       if [ ! -f $LANDMARK ]; then LANDMARK=data/intact_output/${SPB}-${SPA}.tsv; fi
       
       
       
       TRANSFORMEDB_A="mundo2data/TRANSFORMED-${SPB}->${SPA}-${MDSDIM}-${NOLANDMARKS}.npy"
       MDSDISTA_B="mundo2data/MDS-DIST-${SPB}->${SPA}-${MDSDIM}-${NOLANDMARKS}.npy"
       CMD="./run_mundo2.py --ppiA ${PPIA} --ppiB ${PPIB} --nameA ${SPA} --nameB ${SPB} --dsd_A_dist ${DSDA} --dsd_B_dist ${DSDB} --thres_dsd_dist ${THRES_DSD_DIST} --json_A ${JSONA} --json_B ${JSONB} --mds_A ${MDSA} --mds_B ${MDSB} --mds_r ${MDSDIM} --landmarks_a_b ${LANDMARK} --no_landmarks ${NOLANDMARKS} --model ${MODEL} --transformed_b_a ${TRANSFORMEDB_A} --mds_dist_a_b ${MDSDISTA_B} --compute_go_eval --kA ${KA} --kB ${KB} --metrics ${METRICS} --output_file outputs.tsv --go_A ${GOA} --go_B ${GOB} --compute_isorank --wB ${WEIGHTMUNDO}"
       
       # Save the command that is just run
       echo $CMD >> mundo2logs/${LOGPREF}_${SPA}_${SPB}.cmd
       echo "Queuing command: $CMD"
       
       sbatch -o mundo2logs/${LOGPREF}_${SPA}_${SPB}.log --mem 128000 --partition preempt --time=1-10:00:00 --job-name=isorank-${SPA}-${SPB} --partition=preempt $CMD
    done
done

# python run_mundo2.py --ppiA data/intact_output/fly.s.tsv --ppiB data/intact_output/bakers.s.tsv --nameA fly --nameB bakers --dsd_A_dist mundo2data/fly-dsd-dist.npy --dsd_B_dist mundo2data/yeast-dsd-dist.npy --thres_dsd_dist 10 --json_A mundo2data/fly.json --json_B mundo2data/yeast.json --mds_A mundo2data/fly-isomap.npy --mds_B mundo2data/yeast-isomap.npy --mds_r 100 --landmarks_a_b mundo2data/isorank_fly_bakers.tsv --no_landmarks 450 --model "mundo2data/yeast->fly.sav" --transformed_b_a "mundo2data/yeast->fly_emb.npy" --mds_dist_a_b "mundo2data/dist_fly->yeast.npy" --compute_go_eval --kA 10 --kB 10 --metrics top-1-acc --output_file testfly_yeast.tsv --go_A data/go/fly.output.mapping.gaf --go_B data/go/bakers.output.mapping.gaf  # only apply --compute_isorank if you have not generated isorank


# https://www.iscb.org/glbio2023


# python run_mundo2.py --ppiA data/intact_output/fly.s.tsv --ppiB data/intact_output/bakers.s.tsv --nameA fly --nameB bakers --dsd_A_dist mundo2data/fly-dsd-dist.npy --dsd_B_dist mundo2data/yeast-dsd-dist.npy --thres_dsd_dist 10 --json_A mundo2data/fly.json --json_B mundo2data/yeast.json --mds_A mundo2data/fly-isomap.npy --mds_B mundo2data/yeast-isomap.npy --mds_r 100 --landmarks_a_b mundo2data/isorank_fly_bakers.tsv --no_landmarks 450 --model "mundo2data/yeast-fly.sav" --compute_go_eval --kA 10 --kB 10 --metrics top-1-acc --output_file testfly_yeast.tsv --go_A data/go/fly.output.mapping.gaf --go_B data/go/bakers.output.mapping.gaf

# python run_mundo2.py --ppiA data/intact_output/fly.s.tsv --ppiB data/intact_output/bakers_add.s.tsv --nameA fly --nameB bakers_add --dsd_A_dist mundo2data/fly-dsd-dist.npy --dsd_B_dist mundo2data/yeastPlus-dsd-dist.npy --thres_dsd_dist 10 --json_A mundo2data/fly.json --json_B mundo2data/yeastPlus.json --mds_A mundo2data/fly-isomap.npy --mds_B mundo2data/yeastPlus-isomap.npy --mds_r 100 --landmarks_a_b mundo2data/isorank_fly_bakersPlus.tsv --no_landmarks 450 --model "mundo2data/yeastPlus-fly.sav" --compute_go_eval --kA 10 --kB 10 --metrics top-1-acc --output_file testfly_yeastPlus.tsv --go_A data/go/fly.output.mapping.gaf --go_B data/go/bakers.output.mapping.gaf