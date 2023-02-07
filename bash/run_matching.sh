#!/bin/bash 


IP_FOLDER=../data/intact_output/
OP_FOLDER=${IP_FOLDER}/matching_1




if [ ! -d ${OP_FOLDER} ]; then mkdir -p ${OP_FOLDER}; fi

ALPHA=( 0.200000 0.400000 0.600000 0.800000 )
pairs=(fly-bakers fly-mouse fly-rat human-bakers human-fly human-mouse human-rat mouse-bakers mouse-rat rat-bakers)
ITER=20
ALIGN=2000

mkdir -p matching_logs1

for A in ${ALPHA[@]}
do
    for P in ${pairs[@]}
    do
        ORG1=${P%-*}
        ORG2=${P#*-}
        ORG1F=${IP_FOLDER}/${ORG1}.s.tsv
        ORG2F=${IP_FOLDER}/${ORG2}.s.tsv
        # JS1F=${IP_FOLDER}/${ORG1}.json
        # JS2F=${IP_FOLDER}/${ORG2}.json
        PAIR=${IP_FOLDER}/${P}.tsv
        
        OFILE=${OP_FOLDER}/res_${P}_${A}_${ITER}_${ALIGN}.tsv
        sbatch -o matching_logs1/out_${A}_${P}_${ITER}_${ALIGN}.log --mem 64000 --partition preempt --time=1-10:00:00 --job-name=isorank_MATCH --partition=preempt ../src/session.py \
            --net1 ${ORG1F} --net2 ${ORG2F} --rblast ${PAIR} --alpha $A --npairs $ALIGN --niter $ITER \
            --output $OFILE --oboloc=../data/go/go-basic.obo --annot1=../data/go/${ORG1}.output.mapping.gaf --annot2=../data/go/${ORG2}.output.mapping.gaf
    done
done