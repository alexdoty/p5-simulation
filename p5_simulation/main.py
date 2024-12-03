import numpy as np
from p5_simulation.trees import Network, MeterType
from p5_simulation.utils import (
    augment_vector,
    augment_matrix,
    augment_matrices,
    pretty,
    normal_characteristic,
    normal_quantile,
)
from p5_simulation.optimize import greedy_solve
import cmath
import scipy
import math


def main():
    EM = MeterType.EM
    PMU = MeterType.PMU
    net = Network.from_connections(
        [
            # [0,1, EM, 40+2j, 1000+4j],
            # [0,2, EM, 30+3j, 1000+10j],
            # [0,3, 30, 1000],
            # [2,4, 30, 1000],
            # [2,5, 30, 1000],
            # [2,6, 30, 1000],
            # [0, 1, EM, 40 + 40j],
            # [0, 2, MeterType.NONE, 50 + 40j, 2_000 + 1_000j],
            # [0, 3, MeterType.NONE, 50 + 40j, 2_000 + 1_000j],
            # [0, 11, MeterType.NONE, 50 + 40j, 2_000 + 1_000j],
            # [0, 12, MeterType.NONE, 50 + 40j, 2_000 + 1_000j],
            # [0, 13, MeterType.NONE, 50 + 40j, 2_000 + 1_000j],
            # [1, 4, MeterType.NONE, 50 + 40j, 2_000 + 1_000j],
            # [1, 5, MeterType.NONE, 50 + 40j, 2_000 + 1_000j],
            # [2, 6, MeterType.NONE, 50 + 40j, 2_000 + 1_000j],
            # [2, 7, MeterType.NONE, 50 + 40j, 2_000 + 1_000j],
            # [3, 8, MeterType.NONE, 50 + 40j, 2_000 + 1_000j],
            # [3, 9, MeterType.NONE, 50 + 40j, 2_000 + 1_000j],
            # [3, 10, MeterType.NONE, 50 + 40j, 2_000 + 1_000j],
            # [11, 14, MeterType.NONE, 50 + 40j, 2_000 + 1_000j],
            # [11, 15, MeterType.NONE, 50 + 40j, 2_000 + 1_000j],
            # [11, 16, MeterType.NONE, 50 + 40j, 2_000 + 1_000j],
            # [12, 17, MeterType.NONE, 50 + 40j, 2_000 + 1_000j],
            # [12, 18, MeterType.NONE, 50 + 40j, 2_000 + 1_000j],
            # [12, 19, MeterType.NONE, 50 + 40j, 2_000 + 1_000j],
            # [12, 20, MeterType.NONE, 50 + 40j, 2_000 + 1_000j],
            # [12, 21, MeterType.NONE, 50 + 40j, 2_000 + 1_000j],
            # [12, 22, MeterType.NONE, 5.694 + 40j, 2_000 + 1_000j],

            # [0, 1, MeterType.NONE, 31 + 2j],
            # [1, 2, PMU, 20 + 5j, 1_000 + 500j],
            # [1, 3, MeterType.NONE, 12 + 3j, 1_000 + 500j],

            [0, 1, PMU, 12 + 3j, 1_000 + 600j],
            [0, 2, PMU, 4 + 1j, 1_500 + 300j],
            [1, 3, PMU, 5 + 2j, 2_000 + 400j],
            [1, 4, PMU, 9 + 3j, 800 + 100j]
            # [1, 3, EM, 50 + 60j, 2_000 + 3_000j],
            # [1, 4, EM, 50 + 30j, 2_000 + 2_000j],
            # [1, 5, EM, 50 + 30j, 2_000 + 2_000j],
            # [0, 1, 10 + 10j, 1000 + 1000j],
            # [0, 2, 5 + 10j, 1000 + 1000j],
            # [0, 3, 10 + 10j, 1000 + 1000j],
            # [0, 4, 10 + 20j, 2000 + 1000j],
            # [0, 5, 10 + 10j, 3000 + 1000j],
            # [0, 6, 10 + 10j, 5000 + 1000j],
            # [0, 7, 10 + 30j, 500 + 1000j],
            # [0, 8, 10 + 10j, 1000 + 4000j],
            # [0, 9, 10 + 10j, 1000 + 2000j],
            # [0, 10, 40 + 10j, 1000 + 1000j],
            # [0, 11, 10 + 10j, 1000 + 9000j],
            # [0, 12, 10 + 10j, 1000 + 1000j],
            # [0,1, EM, 30+3j],
            # [1,2, EM, 50+3j, 10_000+100j],
            # [1,3, PMU, 40+3j],
            # [3,4,PMU,  10+3j, 200+ 1j],
            # [3,5,EM, 10+1j, 3_500+50j],
            # [3,6,EM, 20+6j, 800+20j],
            # [0,7,PMU, 100+3j, 5_000 + 10_000j],
            # [0,8,PMU, 5+3j],
            # [8,9,PMU, 10+2j],
            # [9,10,EM, 5+3j, 1_000+200j],
            # [9,11,PMU, 5+1j, 1_000+50j],
            # [8,12,PMU, 25+3j, 100+5j],
            # [8,13,EM, 5+0.5j, 4_000+13j],
        ]
    )
    # net.set_angles()


    net.print_node_stats()

    new_net, locs = greedy_solve(net,6)

    print("1-indexed locations", [i + 1 for i in locs])

    return

    D = net.create_D_matrix()

    _ = net.create_C_matrix()

    x = net.state_vector()
    z = D @ net.realize_measurements()

    sigma_1, sigma_2 = net.compute_sigmas()

    A, g = net.compute_A_and_g(z, sigma_1, sigma_2)

    F11 = net.compute_F11_matrix(A)

    g_bar = augment_vector(g)

    x_hat_bar = F11 @ g_bar

    x_hat = x_hat_bar[: net.size * 2]

    print("z", z)
    print("x", x)
    print("x_hat", x_hat)

    F1 = F11[: net.size * 2, : net.size * 2]
    F2 = F11[: net.size * 2, net.size * 2 : net.size * 4]

    idx = 3

    a = F1[idx, idx]
    b = F2[idx, idx]

    c = F1[idx + net.size, idx + net.size]
    d = F2[idx + net.size, idx + net.size]

    print("\nVoltage")
    cov = 0.5 * np.array([[(a + b).real, b.imag], [b.imag, (a - b).real]])
    print("Cov:", cov)

    vals, vecs = np.linalg.eig(cov)

    print("Vals:", vals)
    print("Vecs:", vecs)

    print("Angle:", math.atan2(vecs[1, 0], vecs[0, 0]))

    print("Major:", (vals[0] * scipy.stats.chi2.ppf(0.95, 2)) ** 0.5)
    print("Minor:", (vals[1] * scipy.stats.chi2.ppf(0.95, 2)) ** 0.5)

    print("\nCurrent")
    cov = 0.5 * np.array([[(c + d).real, d.imag], [d.imag, (c - d).real]])
    print("Cov:", cov)

    vals, vecs = np.linalg.eig(cov)

    print("Vals:", vals)
    print("Vecs:", vecs)

    print("Angle:", math.atan2(vecs[1, 0], vecs[0, 0]))
    print((scipy.stats.chi2.ppf(0.95, 2) / scipy.stats.chi2.ppf(0.05, 2)) ** 0.5)
    print("Major:", (vals[0] * scipy.stats.chi2.ppf(0.95, 2)) ** 0.5)
    print("Minor:", (vals[1] * scipy.stats.chi2.ppf(0.95, 2)) ** 0.5)
    # k = A_inv @ A

    # k1 = np.identity(k.shape[0]) - k
    # # np.set_printoptions(threshold=100000, linewidth=10000, edgeitems=30)
    # # print(np.vectorize(lambda x: round(complex(x).real, 2))(k1))

    # residuals = 0
    # runs = 10
    # for _ in range(runs):
    #     z = net.compute_z_vector(D, u_stdev, i_stdev)

    #     g = 2 * D.T @ np.linalg.inv(P) @ z
    #     g_bar = augment_vector(g)

    #     x_hat = (F11 @ g_bar)[: net.size * 2]

    #     # x_hat = np.linalg.lstsq(A, np.concatenate([g_bar, np.zeros(C_bar.shape[0])]))[0][:net.size * 2]

    #     # print(x_hat)
    #     # print(net.root.state_vector())

    #     residuals += np.linalg.norm(x_hat - net.state_vector())
    # print(residuals / runs)

    # net.draw()
    # net.draw()

    # MLE = net.MLE_matrix(D)

    # MLE_inv = np.linalg.inv(MLE)
    # residuals = 0
    # for _ in range(1):

    #     result_vector = net.MLE_result(z, D)

    #     x_hat = (MLE_inv @ result_vector)[: net.size * 2 - 1]
    #     print(x_hat)
    #     print(net.root.state_vector())
    #     residuals += np.linalg.norm(x_hat - net.root.state_vector())

    # print(residuals / 100)
