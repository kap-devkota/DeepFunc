#!/cluster/tufts/cowenlab/.envs/denoise/bin/python
import sys
sys.path.append("./")
import numpy
import glidetools.algorithm.dsd as dsd
import numpy as np
from numpy.linalg import pinv
import json
from typing import List, Tuple, Dict
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.linalg import eig
import argparse
import os
from mundo2.io_utils import compute_adjacency, compute_pairs
from mundo2.isorank import compute_isorank_and_save
from mundo2.predict_score import topk_accs, compute_metric, dsd_func, dsd_func_mundo, scoring_fcn
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
import re
import pandas as pd


# python run_mundo_munk.py --ppiA data/intact_output/fly.s.tsv --ppiB data/intact_output/bakers.s.tsv --nameA fly --nameB bakers --dsd_A_dist mundo2data/fly-dsd-dist.npy --dsd_B_dist mundo2data/yeast-dsd-dist.npy --thres_dsd_dist 10 --json_A mundo2data/fly.json --json_B mundo2data/yeast.json --landmarks_a_b mundo2data/isorank_fly_bakers.tsv --no_landmarks 450 --munk_matrix bakers-fly-munk.npy  --compute_go_eval --kA 10 --kB 10 --metrics top-1-acc --output_file gsterin_testfly_yeast.tsv --go_A data/go/fly.output.mapping.gaf --go_B data/go/bakers.output.mapping.gaf  # only apply --compute_isorank if you have not generated isorank

def getargs():
    parser = argparse.ArgumentParser()

    # PPI
    parser.add_argument("--ppiA", help="PPI for species A: (target)")
    parser.add_argument("--ppiB", help="PPI for species B: (source)")
    parser.add_argument("--nameA", help="Name of species A")
    parser.add_argument("--nameB", help="Name of species B")

    # DSD
    parser.add_argument("--dsd_A_dist", default=None,
                        help="Precomputed DSD distance for Species A. If not present, the matrix will be computed at this location")
    parser.add_argument("--dsd_B_dist", default=None,
                        help="Precomputed DSD distance for Species B. If not present, the matrix will be computed at this location")
    parser.add_argument("--thres_dsd_dist", type=float, default=10,
                        help="If the DSD distance computed is > this threshold, replace that with the threshold")

    # JSON
    parser.add_argument("--json_A", default=None,
                        help="Protein annotation to Matrix index in json format for DSD and MDS matrix of A. If not present, the JSON file will be computed at this location")
    parser.add_argument("--json_B", default=None,
                        help="Protein annotation to Matrix index in json format for DSD and MDS matrix of B. If not present, the JSON file will be computed at this location")

    # ISORANK
    parser.add_argument("--landmarks_a_b", default=None,
                        help="the tsv file with format: `protA  protB  [score]`. Where protA is from speciesA and protB from speciesB")
    parser.add_argument("--compute_isorank", action="store_true", default=False,
                        help="If true, then the landmarks tsv file is a sequence map obtained from Reciprocal BLAST and we need to compute isorank. If false, then the landmark file is a precomputed isorank mapping. This computed isorank matrix is then saved to the file {landmarks_a_b}_{alpha}.tsv")
    parser.add_argument("--isorank_alpha", type=float, default=0.7)
    parser.add_argument("--no_landmarks", type=int, default=500, help="How many ISORANK mappings to use?")

    # MUNK
    parser.add_argument("--munk_matrix", default=None,
                        help="Location of the munk matrix. If not provided, the computed munk matrix is saved at this location")

    # Evaluation
    parser.add_argument("--compute_go_eval", default=False, action="store_true",
                        help="Compute GO functional evaluation on the model?")
    parser.add_argument("--kA", default="10,15,20,25,30", help="Comma seperated values of kA to test")
    parser.add_argument("--kB", default="10,15,20,25,30", help="Comma seperated values of kB to test")
    parser.add_argument("--metrics", default="top-1-acc,", help="Comma separated metrics to test on")
    parser.add_argument("--wB", default=0.4, type=float)
    parser.add_argument("--output_file", help="Output tsv scores file")

    # GO files
    parser.add_argument("--go_A", default=None, help="GO files for species A in TSV format")
    parser.add_argument("--go_B", default=None, help="GO files for species B in TSV format")
    parser.add_argument("--go_h", default="molecular_function,biological_process,cellular_component",
                        help="Which of the three GO hierarchies to use, in comma separated format?")
    return parser.parse_args()


def get_scoring(metric, all_go_labels = None, **kwargs):
    acc = re.compile(r'top-([0-9]+)-acc')
    match_acc = acc.match(metric)
    if match_acc:
        k = int(match_acc.group(1))

        def score(prots, pred_go_map, true_go_map):
            return topk_accs(prots, pred_go_map, true_go_map, k=k)

        return score
    else:
        if metric == "aupr":
            met = average_precision_score
        elif metric == "auc":
            met = roc_auc_score
        elif metric == "f1max":
            def f1max(true, pred):
                pre, rec, _ = precision_recall_curve(true, pred)
                f1 = (2 * pre * rec) / (pre + rec + 1e-7)
                return np.max(f1)
            met = f1max
        sfunc = scoring_fcn(all_go_labels, met, **kwargs)
    return sfunc

def compute_dsd_dist(ppifile, dsdfile, jsonfile, threshold=-1, **kwargs):
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
        ppdf = pd.read_csv(ppifile, sep="\t", header=None)
        Ap, protmap = compute_adjacency(ppdf)
        if jsonfile is not None:
            with open(jsonfile, "w") as jf:
                json.dump(protmap, jf)
        DSDemb = dsd.compute_dsd_embedding(Ap, is_normalized=False)
        DSDdist = squareform(pdist(DSDemb))
        if dsdfile is not None:
            np.save(dsdfile, DSDdist)
        if threshold > 0:
            DSDdist = np.where(DSDdist > threshold, threshold, DSDdist)
        return DSDdist, protmap

def compute_munk(src_dsd: np.ndarray,
                 tar_dsd: np.ndarray,
                 src_node_map: Dict[str, int],
                 tar_node_map: Dict[str, int],
                 landmarks: List[Tuple[str, str]]) -> np.ndarray:
    """
    Given source dsd D_A, target dsd D_B, D_A = S_A @ S_A, D_B = S_B @ S_B. Landmarks := L
    1) Get S_A
    2) \hat{T}_B = (S_A[L])^{\dagger} @ S_B[L] @ S_B => m x l @ l x n @  n x n = m x n
           motive:
           S_B[L] = S_A[L] K, K = S_A[L]^{\dagger} S_B[L]
           T_{A->B} = (S_A K) = S_A S_A[L]^{\dagger} S_B[L]
    So, we should transform source to target.
    3) M = S_A @ (S_A[L])^\dagger @ S_B[L] @ S_B
           
    """
    def rkhs(matrix: np.ndarray)-> np.ndarray:
        def is_symmetric(a: np.ndarray, rtol: float=1e-05, atol: float=1e-08) -> bool:
            return np.allclose(a, a.T, rtol=rtol, atol=atol)

        if not is_symmetric(matrix):
            print('ERROR: Cannot embed asymmetric kernel into RKHS. Closing...')
            exit()

        eigenvalues, eigenvectors = eig(matrix)
        return eigenvectors @ np.diag(np.sqrt(eigenvalues))

    def embed_matrices(source_rkhs: np.ndarray, # Source : m x m 
                       target_diff: np.ndarray, # Target : n x n
                       landmark_id: List[Tuple[int, int]]) -> np.ndarray:
        s_landmark_id, t_landmark_id = zip(*landmark_id)
        return pinv(source_rkhs[s_landmark_id, :]) @ (target_diff[t_landmark_id, :]) # (SA_L)^{\dagger} @ SB_L @ SB
    # (m x L) @ (L x n) => m x n

    landmark_indices = [(src_node_map[landmark[0]], tar_node_map[landmark[1]]) for landmark in landmarks]
    print('Computing RKHS for source network... ')
    source_rkhs = rkhs(src_dsd)
    print('Embedding matrices... ')
    target_rkhs_hat = embed_matrices(source_rkhs, tar_dsd, landmark_indices)
    print('Creating final munk matrix... ')
    munk_matrix = np.dot(source_rkhs, target_rkhs_hat).T # ((m x m) x (m x n)).T => n x m
    return munk_matrix
# Target is moved to source and then we compute the dot product


def get_go_maps(nmap, gofile, gotype):
    """
    Get the GO maps
    """
    df = pd.read_csv(gofile, sep="\t")
    df = df.loc[df["type"] == gotype]
    gomaps = df.loc[:, ["GO", "swissprot"]].groupby("swissprot", as_index=False).aggregate(list)
    gomaps = gomaps.values
    go_outs = {}
    all_gos = set()
    for prot, gos in gomaps:
        if prot in nmap:
            all_gos.update(gos)
            go_outs[nmap[prot]] = set(gos)
    for i in range(len(nmap)):
        if i not in go_outs:
            go_outs[i] = {}
    return go_outs, all_gos


def main(args):
    """
    Main function
    """
    DSDA, nmapA = compute_dsd_dist(args.ppiA, args.dsd_A_dist, args.json_A, threshold=10,
                                   msg="Running DSD distance for Species A")
    DSDB, nmapB = compute_dsd_dist(args.ppiB, args.dsd_B_dist, args.json_B, threshold=10,
                                   msg="Running DSD distance for Species B")

    if args.compute_isorank:
        isorank_file = f"{args.nameA}_{args.nameB}_isorank_alpha_{args.isorank_alpha}.tsv"
        compute_isorank_and_save(args.ppiA, args.ppiB, args.nameA, args.nameB,
                                 matchfile=args.landmarks_a_b,
                                 alpha=args.isorank_alpha,
                                 n_align=args.no_landmarks, save_loc=isorank_file,
                                 msg="Running ISORANK.")
    else:
        isorank_file = args.landmarks_a_b

    if args.munk_matrix is not None and os.path.exists(args.munk_matrix):
        munk = np.load(args.munk_matrix)
    else:
        landmarks_df = pd.read_csv(isorank_file, sep='\t')
        landmarks_df = landmarks_df[[args.nameB, args.nameA]]
        munk = compute_munk(DSDB, DSDA, nmapB, nmapA, landmarks_df.values.tolist()[:args.no_landmarks])
        np.save(args.munk_matrix, munk)

    results = []
    "settings: nameA, nameB, landmark, gotype, topkacc, dsd/mundo?, kA, kB, "

    settings = [args.nameA, args.nameB, args.no_landmarks]
    if args.compute_go_eval:
        """
        Perform evaluations
        """
        kAs = [int(k) for k in args.kA.split(",")]
        kBs = [int(k) for k in args.kB.split(",")]
        gos = args.go_h.split(",")
        gomapsA = {}
        gomapsB = {}
        for go in gos:
            gomapsA[go], golabelsA = get_go_maps(nmapA, args.go_A, go)
            gomapsB[go], golabelsB = get_go_maps(nmapB, args.go_B, go)
            golabels = golabelsA.union(golabelsB)
            for metric in args.metrics.split(","):
                score = get_scoring(metric, golabels)
                for kA in kAs:
                    settings_dsd = settings + [go, metric, "dsd-knn", kA, -1]
                    scores, _ = compute_metric(dsd_func(DSDA, k=kA), score, list(range(len(nmapA))), gomapsA[go],
                                               kfold=5)
                    print(f"GO: {go}, DSD, k: {kA} ===> {np.average(scores):0.3f} +- {np.std(scores):0.3f}")
                    settings_dsd += [np.average(scores), np.std(scores)]
                    results.append(settings_dsd)
                    for kB in kBs:
                        settings_mundo = settings + [go, metric, f"mundo-munk-knn-weight-{args.wB:0.3f}", kA, kB]
                        scores, _ = compute_metric(
                            dsd_func_mundo(DSDA, munk, gomapsB[go], k=kA, k_other=kB, weight_other=args.wB),
                            score, list(range(len(nmapA))), gomapsA[go], kfold=5)
                        settings_mundo += [np.average(scores), np.std(scores)]

                        print(
                            f"GO: {go}, MUNDO-MUNK, kA: {kA}, kB: {kB} ===> {np.average(scores):0.3f} +- {np.std(scores):0.3f}")
                        results.append(settings_mundo)
        columns = ["Species A", "Species B", "Landmark no", "GO type", "Scoring metric",
                   "Prediction method",
                   "kA", "kB", "Average score", "Standard deviation"]
        resultsdf = pd.DataFrame(results, columns=columns)
        resultsdf.to_csv(args.output_file, sep="\t", index=None, mode="a", header=not os.path.exists(args.output_file))
    return


if __name__ == "__main__":
    main(getargs())

