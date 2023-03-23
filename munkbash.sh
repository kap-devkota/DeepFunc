#!/bin/bash

SPECIESA=(rat)
FILESA=(data/intact_output/rat.s.tsv data/intact_output/human_sub.s.tsv)

# SPECIESB=(bakers fly mouse)
SPECIESB=(bakers fly mouse)
FILESB=(data/intact_output/bakers.s.tsv data/intact_output/fly.s.tsv  data/intact_output/mouse.s.tsv)

KA="10,50"
KB="0,50"

THRES_DSD_DIST=10
METRICS="top-1-acc,f1max"

OPFILE="outputs/outputs_munk.tsv"

NOLANDMARKS=500
WEIGHTMUNDO=0.66

# (500, 0.2), (500, 0.6), (500, 0.8), (500, 1)

# (200, x), (300, x), (400, x), (100, x), (750, x)
MODE=
while getopts "l:w:d" args
do
    case $args in 
        l) NOLANDMARKS=${OPTARG}
        ;;
        w) WEIGHTMUNDO=${OPTARG}
        ;;
        d) KA="10"; KB="10";METRICS="f1max";MODE="-DEBUG";NOLANDMARKS=50;OPFILE="outputs${MODE}.tsv"
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
       MUNK=mundo2data/MUNK_${SPA}_${SPB}.npy

       LANDMARK=data/intact_output/${SPA}-${SPB}.tsv
       if [ ! -f $LANDMARK ]; then LANDMARK=data/intact_output/${SPB}-${SPA}.tsv; fi
              
       CMD="./run_mundo_munk.py --ppiA ${PPIA} --ppiB ${PPIB} --nameA ${SPA} --nameB ${SPB} --dsd_A_dist ${DSDA} --dsd_B_dist ${DSDB} --thres_dsd_dist ${THRES_DSD_DIST} --json_A ${JSONA} --json_B ${JSONB} --landmarks_a_b ${LANDMARK} --no_landmarks ${NOLANDMARKS} --munk_matrix $MUNK --compute_go_eval --kA ${KA} --kB ${KB} --metrics ${METRICS} --output_file ${OPFILE} --go_A ${GOA} --go_B ${GOB} --compute_isorank --wB ${WEIGHTMUNDO}"
       
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
done

# python run_mundo2.py --ppiA data/intact_output/fly.s.tsv --ppiB data/intact_output/bakers.s.tsv --nameA fly --nameB bakers --dsd_A_dist mundo2data/fly-dsd-dist.npy --dsd_B_dist mundo2data/yeast-dsd-dist.npy --thres_dsd_dist 10 --json_A mundo2data/fly.json --json_B mundo2data/yeast.json --mds_A mundo2data/fly-isomap.npy --mds_B mundo2data/yeast-isomap.npy --mds_r 100 --landmarks_a_b mundo2data/isorank_fly_bakers.tsv --no_landmarks 450 --model "mundo2data/yeast->fly.sav" --transformed_b_a "mundo2data/yeast->fly_emb.npy" --mds_dist_a_b "mundo2data/dist_fly->yeast.npy" --compute_go_eval --kA 10 --kB 10 --metrics top-1-acc --output_file testfly_yeast.tsv --go_A data/go/fly.output.mapping.gaf --go_B data/go/bakers.output.mapping.gaf  # only apply --compute_isorank if you have not generated isorank


# https://www.iscb.org/glbio2023
