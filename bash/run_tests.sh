#!/bin/bash 

ITER=20
ALPHA=0.6
while getopts "a:i:" arg
do
    case ${arg} in
        a) ALPHA=${OPTARG}
        ;;
        i) ITER=${OPTARG}
        ;;
    esac
done

pairs=(fly-bakers fly-mouse fly-rat human-bakers human-fly human-mouse human-rat mouse-bakers mouse-rat rat-bakers)
FOLDER=../data/intact_output/working

for p in ${pairs[@]}
do 
    p2=${p#*-}
    p1=${p%-*}
    p1file=$FOLDER/$p1
    p2file=$FOLDER/$p2
    pfile=$FOLDER/${p}.tsv
    
    if [ ! -f $pfile ]; then echo "Cannot find the pair file"; exit 1; fi
    
    ofold=$FOLDER/outputs
    matlab -nodisplay -nodesktop -nosplash -r "addpath('../src');run_tests('$p1file', '$p2file', '$pfile', $ALPHA, $ITER, '$ofold'); exit;"
done
