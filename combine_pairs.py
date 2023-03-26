#!/cluster/tufts/cowenlab/.envs/denoise/bin/python
import sys
sys.path.append("./")
import argparse
import os

K = 15


def getargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", help = "name of the species")
    parser.add_argument('-t', type = float, help = 'threshold value', default=0.5)
    parser.add_argument('-k', help = 'knn value to choose')
    parser.add_argument('-m', help = 'invert mode', default=False)
    
    return parser.parse_args()


def main(args):
    
    pairs = set([tuple(sorted((x, y))) for [x, y] in (f.split('\t') for f in open(f'data/intact_output/{args.s}.s.tsv', 'r').read().splitlines())])
    query = set([tuple(sorted((x, y))) for [x, y, _] in (f.split('\t') for f in open(f'data/pairs/{args.s}_{args.k}.tsv', 'r').read().splitlines())])
    
    thresh = (lambda x: x <= args.t) if args.m else lambda x: x >= args.t

    validPairs = set()
    
    for [x, y, t] in [f.split('\t') for f in open(f'data/output/{args.s}_{K}.tsv', 'r').read().splitlines()]:
        pair = tuple(sorted((x, y)))
        if pair in query and thresh(float(t)):
            validPairs.add(pair)
            
    if args.m:
        out = pairs.difference(validPairs)
    else:
        out = pairs.union(validPairs)
        
        
    print(f'Original Set: {len(pairs)}\tQueried Set: {len(query)}\t Positive Set: {len(validPairs)}\tFinal Set: {len(out)}')
                        
    with open(f'data/intact_output/{args.s}_{args.k}.s.tsv', 'w') as newF:
        newF.writelines([f'{x}\t{y}\n' for (x, y) in sorted(out)])
    
if __name__ == "__main__":
    main(getargs())