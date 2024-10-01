import numpy as np
from p5_simulation.trees import NetworkNode
from p5_simulation.utils import *


def main():
    root = NetworkNode(None, [], None)
    sub = NetworkNode(root, [], None)
    house1 = NetworkNode(root, [], 2_000.0)
    house2 = NetworkNode(sub, [], 3_000.0)
    house3 = NetworkNode(sub, [], 500.0)
    root.add_child(sub, 40.0)
    root.add_child(house1, 50.0)
    sub.add_child(house2, 10.0)
    sub.add_child(house3, 10.0)

    # print("Root:")
    # root.print_stats()
    # print("House 1:")
    # house1.print_stats()
    # print("House 2:")
    # house2.print_stats()

    total_nodes = root.set_node_indices(0) + 1
    # print(root.all_equations(total_nodes))

    # x = root.state_vector(total_nodes)
    # print(f"x vector: {x}", end="\n\n")

    D = create_D_matrix([2, 3, 5], total_nodes)

    MLE = MLE_matrix(root, D, total_nodes)

    MLE_inv = np.linalg.inv(MLE)
    residuals = 0
    for _ in range(1):
        z = compute_z_vector(root, D, 4.0, 0.002, total_nodes)

        result_vector = MLE_result(root, z, D, total_nodes)

        x_hat = (MLE_inv @ result_vector)[: total_nodes * 2]
        print(x_hat)
        print(root.state_vector(total_nodes))
        residuals += np.linalg.norm(x_hat - root.state_vector(total_nodes))

    print(residuals / 100)
