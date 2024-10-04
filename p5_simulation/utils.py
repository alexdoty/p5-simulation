from numpy.typing import NDArray
import numpy as np
# from p5_simulation.trees import NetworkNode
#
def augment_vector(vec: NDArray) -> NDArray:
    return np.concatenate((vec, vec.conj()))

def augment_matrix(mat: NDArray) -> NDArray:
    return augment_matrices(mat, np.zeros_like(mat))

def augment_matrices(m1: NDArray, m2: NDArray) -> NDArray:
    return np.block([[m1, m2], [m2.conj() , m1.conj()]])


