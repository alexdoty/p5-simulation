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

def pretty(c: complex):
    m = max(c.real, c.imag)

    if m < 1:
        unit = "m"
        f = 0.001
    elif m < 1_000:
        unit = ""
        f = 1
    elif m < 1_000_000:
        unit = "k"
        f = 1_000
    else:
        unit = "M"
        f = 1_000_000
    return f"({round_rel(c.real / f, 3)} + {round_rel(c.imag / f, 3)}i) {unit}Ω"

def round_rel(x: float, digits: int) -> str:
    s = str(x)
    if len(s) <= digits:
        return s
    elif "." in s[:digits]:
        return s[:digits + 1]
    else:
        return s[:digits]
