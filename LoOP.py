import numpy as np
from scipy.special import erf   
import pandas as pd

from utils import k_nearest_neighbors, plot_results

def LoOP(X: np.ndarray, lmbda: float, k: int) -> np.ndarray:
    """
    The matrix X has dimensions n * d, representing samples and dimensions
    respectively .
    We used only scipy.special.erf, numpy
    """
    k_nearset, D = k_nearest_neighbors(X, k)
    sigma = get_sigma(D, k_nearset)
    pdist = lmbda * sigma
    plof = PLOF(pdist, k_nearset)
    ratio = plof / nPLOF(plof, lmbda)
    return np.maximum(erf(ratio / np.sqrt(2)), 0)

def get_sigma(D: np.ndarray, k_nearset: np.ndarray) -> np.ndarray:
    """
    The matrix D has dimensions n * n, representing the distance between each pair of samples
    The matrix k_nearset has dimensions n * k, representing the k nearest neighbors for each sample
    """
    n, k = k_nearset.shape
    indices = np.tile(np.arange(n)[:, None], k)
    return np.sqrt(np.sum(D[indices, k_nearset] ** 2, axis=1) / k)

def PLOF(pdist: np.ndarray, k_nearset: np.ndarray) -> np.ndarray:
    """
    The matrix pdist has dimensions n * 1, representing the density of the point
    The matrix k_nearset has dimensions n * k, representing the k nearest neighbors for each sample
    """
    return pdist / np.mean(pdist[k_nearset], axis=1) - 1

def nPLOF(plof: np.ndarray, lmbda: float) -> np.ndarray:
    """
    The matrix plof has dimensions n * 1
    """
    return lmbda * np.sqrt(np.mean(plof ** 2))



if __name__ == "__main__":
    data = pd.read_csv("data.csv", header=None)
    X = data.iloc[:, :].values
    y = LoOP(X, 0.90, 10)
    plot_results(X, y)