#!/cluster/tufts/cowenlab/.envs/denoise/bin/python
import argparse
import json
import os
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from numpy.linalg import pinv
from scipy.linalg import eig


def rkhs(matrix: np.ndarray)-> np.ndarray:
    def is_symmetric(a: np.ndarray, rtol: float=1e-05, atol: float=1e-08) -> bool:
        return np.allclose(a, a.T, rtol=rtol, atol=atol)

    if not is_symmetric(matrix):
        print('ERROR: Cannot embed asymmetric kernel into RKHS. Closing...')
        exit()

    eigenvalues, eigenvectors = eig(matrix)
    return eigenvectors @ np.diag(np.sqrt(eigenvalues))


def embed_matrices(source_rkhs: np.ndarray,
                   target_diff: np.ndarray,
                   landmark_id: List[Tuple[int, int]]) -> np.ndarray:
    s_landmark_id, t_landmark_id = zip(*landmark_id)
    return pinv(source_rkhs[s_landmark_id, :]) @ (target_diff[t_landmark_id, :])


def coembed_networks(src_dsd: np.ndarray,
                     tar_dsd: np.ndarray,
                     src_node_map: Dict[str, int],
                     tar_node_map: Dict[str, int],
                     landmarks: List[Tuple[str, str]]) -> np.ndarray:
    landmark_indices = [(src_node_map[landmark[0]], tar_node_map[landmark[1]]) for landmark in landmarks]
    print('Computing RKHS for source network... ')
    source_rkhs = rkhs(src_dsd)
    print('Embedding matrices... ')
    target_rkhs_hat = embed_matrices(source_rkhs, tar_dsd, landmark_indices)
    print('Creating final munk matrix... ')
    munk_matrix = np.dot(source_rkhs, target_rkhs_hat)
    return munk_matrix.T


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name_src", default=None,
                        help="Name of the source organism")
    parser.add_argument("--name_tar", default=None,
                        help="Name of the target organism")
    parser.add_argument("--dsd_src", default=None,
                        help="Precomputed DSD distance for source species")
    parser.add_argument("--dsd_tar", default=None,
                        help="Precomputed DSD distance for target species")
    parser.add_argument("--json_src", default=None,
                        help="Protein annotation to Matrix index in json format for DSD matrix of the source organism")
    parser.add_argument("--json_tar", default=None,
                        help="Protein annotation to Matrix index in json format for DSD matrix of the target organism")
    parser.add_argument("--landmarks", default=None,
                        help="File with landmark proteins found by reciprocal BLASTP or Isorank")
    parser.add_argument("--no_landmarks", default=None, help="Number of landmarks to use for coembedding")
    parser.add_argument("--munk_file", default=None, help="Computed munk coembedding file will be saved at this location")
    args = parser.parse_args()

    assert(os.path.exists(args.dsd_src)
           and os.path.exists(args.dsd_tar)
           and os.path.exists(args.json_src)
           and os.path.exists(args.json_tar)
           and os.path.exists(args.landmarks))

    dsd_src = np.load(args.dsd_src)
    with open(args.json_src, "r") as jf:
        nodemap_src = json.load(jf)

    dsd_tar = np.load(args.dsd_tar)
    with open(args.json_tar, "r") as jf:
        nodemap_tar = json.load(jf)

    landmarks_df = pd.read_csv(args.landmarks, sep='\t')
    landmarks_df = landmarks_df[[args.name_src, args.name_tar]]
    munk = coembed_networks(dsd_src, dsd_tar, nodemap_src, nodemap_tar, landmarks_df.values.tolist()[:int(args.no_landmarks)])
    np.save(args.munk_file, munk)

#run_munk_coembedding.py --name_src bakers --name_tar fly --dsd_src ./mundo2data/DSD-DIST-bakers.npy --dsd_tar ./mundo2data/DSD-DIST-fly.npy --json_src ./mundo2data/bakers.json --json_tar ./mundo2data/fly.json --landmarks ./fly_bakers_isorank_alpha_0.7.tsv --no_landmarks 500 --munk_file ./bakers-fly-munk-500
