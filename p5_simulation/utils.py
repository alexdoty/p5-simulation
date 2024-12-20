from numpy.typing import NDArray
import numpy as np
import math
from scipy.special import erfinv


# from p5_simulation.trees import NetworkNode
#
def augment_vector(vec: NDArray) -> NDArray:
    return np.concatenate((vec, vec.conj()))


def augment_matrix(mat: NDArray) -> NDArray:
    return augment_matrices(mat, np.zeros_like(mat))


def augment_matrices(m1: NDArray, m2: NDArray) -> NDArray:
    return np.block([[m1, m2], [m2.conj(), m1.conj()]])

def augment_transformation(size: int) -> NDArray:
    return np.block([
        [np.identity(size), 1j * np.identity(size)],
        [np.identity(size), -1j * np.identity(size)]
    ])


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
        return s[: digits + 1]
    else:
        return s[:digits]


def normal_characteristic(std_dev: float, t: float) -> float:
    return math.exp(-1 / 2 * std_dev**2 * t**2)


def normal_cdf(x: float, stdev: float) -> float:
    return 1 / 2 * (1 + math.erf(x / (2**0.5 * stdev)))


def normal_quantile(p: float, stdev: float) -> float:
    return stdev * 2**0.5 * erfinv(2 * p - 1)
