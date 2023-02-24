import scipy as sp
import numpy as np
from scipy import linalg



def compute_k_svd(M, k):
    U, s, _ = linalg.svd(M)
    s = np.sqrt(s[:k])
    U = U[:, :k]
    return np.multiply(U, s)