import sys
sys.path.append("./")
import argparse
import os
import numpy as np
import pandas as pd
import json
from mundo2.io_utils import compute_adjacency
import glidetools.algorithm.dsd as dsd
from scipy.spatial.distance import squareform, pdist
from sklearn.neighbors import NearestNeighbors

def getargs():
    parser = argparse.ArgumentParser()
    
    # PPI
    parser.add_argument("--ppi", help = "PPI for species A: (target)")
    parser.add_argument("--name", help = "Name of species A")
    
    # DSD
    parser.add_argument("--dsd", default = None, help = "Precomputed DSD distance for Species A. If not present, the matrix will be computed at this location")
    parser.add_argument("--thres_dsd_dist", type = float, default = 10, help = "If the DSD distance computed is > this threshold, replace that with the threshold")
    
    # JSON
    parser.add_argument("--json", default = None, help = "Protein annotation to Matrix index in json format for DSD and MDS matrix of A. If not present, the JSON file will be computed at this location")
    
    # k
    parser.add_argument("--num_neighbors", default=10, help = 'Number of neighbors to get for each protein')
    
    return parser.parse_args()

def compute_dsd_dist(ppifile, dsdfile, jsonfile, threshold = -1, **kwargs):
    assert (os.path.exists(ppifile)) or (os.path.exists(dsdfile) and os.path.exists(jsonfile))
    print(f"[!] {kwargs['msg']}")
    if dsdfile is not None and os.path.exists(dsdfile):
        print(f"[!!] \tAlready computed!")
        DSDdist = np.load(dsdfile)
        if threshold > 0:
            DSDdist = np.where(DSDdist > threshold, threshold, DSDdist)
        with open(jsonfile, "r") as jf:
            protmap = json.load(jf)
        return DSDdist, protmap
    else:
        ppdf = pd.read_csv(ppifile, sep = "\t", header = None)
        Ap, protmap = compute_adjacency(ppdf)
        if jsonfile is not None:
            with open(jsonfile, "w") as jf:
                json.dump(protmap, jf)                
        DSDemb = dsd.compute_dsd_embedding(Ap, is_normalized = False)
        DSDdist = squareform(pdist(DSDemb))
        if dsdfile is not None:
            np.save(dsdfile, DSDdist)
        if threshold > 0:
            DSDdist = np.where(DSDdist > threshold, threshold, DSDdist)
        return DSDdist, protmap

def print_pairings(nn, nmap, sname):
    k = nn.shape[1]
    fname = f'data/pairs/{sname}_{k}.tsv'
    print(f"[!] Printing knn pairs to {fname}")
    
    valid_ids = set(open(f'data/mappings/{sname}_trimmed_ids.tsv', 'r').read().splitlines())
    
    rmap = {val: key for key, val in nmap.items()}
    pairs = set()
    for i in range(nn.shape[0]):
        for j in nn[i]:
            if (j, i) not in pairs:
                pairs.add((i, j))
    with open(fname, 'w') as f:
        for (i, j) in pairs:
            if rmap[i] in valid_ids and rmap[j] in valid_ids:
                f.write(f'{rmap[i]}\t{rmap[j]}\t0\n')
        
def main(args):
    DSD, nmap = compute_dsd_dist(args.ppi, args.dsd, args.json, threshold = 10,
                                  msg = "Running DSD distance for Species A")
    
    print(f"[!] Getting {args.num_neighbors} nearest neighbors")
    knn_distance_based  = NearestNeighbors(n_neighbors=int(args.num_neighbors)).fit(DSD)
    knn = knn_distance_based.kneighbors(return_distance=False)
    
    print_pairings(knn, nmap, args.name)
    
if __name__ == "__main__":
    main(getargs())