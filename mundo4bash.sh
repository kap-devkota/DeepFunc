#!/bin/bash

SPECIESA=(bakers rat)
FILESA=(data/intact_output/bakers.s.tsv data/intact_output/rat.s.tsv)


SPECIESB=(fly)
FILESB=(data/intact_output/fly.s.tsv data/intact_output/mouse.s.tsv)

K=("" "_5" "_10" "_15")

KA="10,50"
KB="20,100"

THRES_DSD_DIST=10
METRICS="top-1-acc,f1max"


SVDDIM=250
NOLANDMARKS=50
WEIGHTMUNDO=0.660
OPFILE=outputs/outputs_m4.tsv
NEPOCH=100

LOGPREF=$(date | tr ' ' '-' | awk '{printf "log-%s",$0}')

for i in $(seq 0 $((${#SPECIESB[@]} - 1)))
do
    for j in $(seq 0 $((${#SPECIESA[@]} - 1)))
    do
      for k in $(seq 0 $((${#K[@]} - 1)))
      do
         EXT=${K[k]}
         SPA=${SPECIESA[j]}
         SPB=${SPECIESB[i]}
         PPIA=data/intact_output/${SPA}${EXT}.s.tsv
         PPIB=data/intact_output/${SPB}${EXT}.s.tsv
         DSDA=mundo2data/DSD-DIST-${SPA}${EXT}.npy
         DSDB=mundo2data/DSD-DIST-${SPB}${EXT}.npy
         JSONA=mundo2data/${SPA}.json
         JSONB=mundo2data/${SPB}.json
         GOA=data/go/${SPA}.output.mapping.gaf 
         GOB=data/go/${SPB}.output.mapping.gaf 
         SVDA=mundo2data/SVD-${SPA}${EXT}-${SVDDIM}.npy
         SVDB=mundo2data/SVD-${SPB}${EXT}-${SVDDIM}.npy
         MODEL="mundo2data/SVD-MODEL-${SPB}-${SPA}-${SVDDIM}-${NOLANDMARKS}${MODE}.sav"
         
         LANDMARK=data/intact_output/${SPA}-${SPB}.tsv
         if [ ! -f $LANDMARK ]; then LANDMARK=data/intact_output/${SPB}-${SPA}.tsv; fi
         
         #TRANSFORMEDB_A="mundo2data/SVD-TRANSFORMED-${SPB}-${SPA}-${SVDDIM}-${NOLANDMARKS}${MODE}.npy"
         SVDDISTA_B="mundo2data/SVD-DIST-${SPA}-${SPB}-${SVDDIM}-${NOLANDMARKS}${MODE}.npy"
         CMD="python run_mundo4.py --ppiA ${PPIA} --ppiB ${PPIB} --nameA ${SPA} --nameB ${SPB} --dsd_A_dist ${DSDA} --dsd_B_dist ${DSDB} --thres_dsd_dist ${THRES_DSD_DIST} --json_A ${JSONA} --json_B ${JSONB} --svd_A ${SVDA} --svd_B ${SVDB} --svd_r ${SVDDIM} --landmarks_a_b ${LANDMARK} --no_landmarks ${NOLANDMARKS} --model ${MODEL} --svd_dist_a_b ${SVDDISTA_B} --compute_go_eval --kA ${KA} --kB ${KB} --metrics ${METRICS} --output_file $OPFILE --go_A ${GOA} --go_B ${GOB} --no_epoch ${NEPOCH} --compute_isorank --wB ${WEIGHTMUNDO} --compute_dsd"
         
         # Save the command that is just run
      #    echo $CMD >> mundo2logs/${LOGPREF}_${SPA}_${SPB}.cmd
         echo "Queuing command: $CMD" >> temp
         # $CMD > mundo2logs/b4_${SPA}_${SPB}.log
         #  if [ -z $MODE ]
         #  then
         #       sbatch -o mundo2logs/baseline_${SPA}_${SPB}.log --mem 128000 --partition preempt --time=1-10:00:00 --job-name=isorank-${SPA}-${SPB} --partition=preempt $CMD
         #   else
         #        $CMD
         #        exit
         #   fi
      done
   done
done

# python run_mundo2.py --ppiA data/intact_output/fly.s.tsv --ppiB data/intact_output/bakers.s.tsv --nameA fly --nameB bakers --dsd_A_dist mundo2data/fly-dsd-dist.npy --dsd_B_dist mundo2data/yeast-dsd-dist.npy --thres_dsd_dist 10 --json_A mundo2data/fly.json --json_B mundo2data/yeast.json --mds_A mundo2data/fly-isomap.npy --mds_B mundo2data/yeast-isomap.npy --mds_r 100 --landmarks_a_b mundo2data/isorank_fly_bakers.tsv --no_landmarks 450 --model "mundo2data/yeast-fly.sav" --transformed_b_a "mundo2data/yeast-fly_emb.npy" --mds_dist_a_b "mundo2data/dist_fly-yeast.npy" --compute_go_eval --kA 10 --kB 10 --metrics top-1-acc --output_file testfly_yeast.tsv --go_A data/go/fly.output.mapping.gaf --go_B data/go/bakers.output.mapping.gaf  # only apply --compute_isorank if you have not generated isorank


# https://www.iscb.org/glbio2023
