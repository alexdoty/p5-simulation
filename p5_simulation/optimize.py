
import numpy as np
from p5_simulation.utils import augment_matrices, augment_matrix, augment_transformation
from p5_simulation.trees import MeterType, Network

def setup(net: Network, k: int) -> list[int]:
    assert k <= 2 * net.size, "k must be less than or equal to the sum of nodes and edges!"
    locations = list(range(0, net.size * 2))

    # Ensure each node has a PMU
    for node in net.nodes:
        node.meter = MeterType.PMU

    return locations


def greedy_solve(net: Network, k: int) -> tuple[Network, list[int]]:
    locations = setup(net, k)
    # assert k <= 2 * net.size, "k must be less than or equal to the sum of nodes and edges!"
    # locations = list(range(0, net.size * 2))

    # # Ensure each node has a PMU
    # for node in net.nodes:
    #     node.meter = MeterType.PMU

    # Compute H, multiply by T^(-1) on the left and T on the right and get the diagonal entries
    # Find the index with the smallest value
    # Remove that measurement and calculate again
    num_locs = net.size * 2
    for i in range(k):
        D = net.create_D_matrix()
        S1, S2 = net.compute_true_sigmas()

        # Compute relevant matrices
        A = net.compute_A(S1, S2)
        F11 = net.compute_F11_matrix(A)
        B = np.linalg.inv(S1 - S2.conjugate() @ np.linalg.inv(S1) @ S2)
        W = S2 @ np.linalg.inv(S1.conjugate())
        J_aug = augment_matrices(
           D.T @ B,
           -D.T @ B @ W
        )
        D_aug = augment_matrix(D)
        H = D_aug @ F11 @ J_aug
        print(D.shape)
        print(H.shape)

        T = augment_transformation(D.shape[0])
        H_real = np.linalg.inv(T) @ H @ T

        # Get arg min of leverages
        loc = 0
        lowest = H_real[0,0] + H_real[num_locs,num_locs]
        for j in range(1, H_real.shape[1] // 2):
            if H_real[j, j] + H_real[j+num_locs, j+num_locs] < lowest:
                loc = j

        if loc >= num_locs:
            print(H_real.shape)
            print(locations)
            print("Broken")
            raise ValueError
        print("Before", loc)
        loc = locations.pop(loc)
        print("After", loc)

        # Check if removing voltage meter or current meter
        if loc < net.size:
            # Remove voltage meter
            match net.nodes[loc].meter:
                case MeterType.PMU:
                    net.nodes[loc].meter = MeterType.CURRENT
                case MeterType.VOLTAGE:
                    net.nodes[loc].meter = MeterType.NONE
        elif loc >= net.size:
            loc -= net.size
            # Remove current meter
            match net.nodes[loc].meter:
                case MeterType.PMU:
                    net.nodes[loc].meter = MeterType.VOLTAGE
                case MeterType.CURRENT:
                    net.nodes[loc].meter = MeterType.NONE
        num_locs -= 1

    return net, locations

def better_solve(net: Network) -> tuple[Network, list[int]]:
    return (Network(), [2, 3])
