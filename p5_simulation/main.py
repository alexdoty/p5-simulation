import numpy as np
from p5_simulation.trees import Network


def main():
    net = Network.from_connections(
        [
            [0, 1, 40],
            [0, 2, 50, 2_000],
            [1, 3, 50, 2_000],
            [1, 4, 50, 2_000],
        ]
    )

    net.print_node_stats()

    D = net.create_D_matrix([0,1,2, 3, 5])

    MLE = net.MLE_matrix(D)

    MLE_inv = np.linalg.inv(MLE)
    residuals = 0
    for _ in range(1):
        z = net.compute_z_vector(D, 4.0, 0.002)

        result_vector = net.MLE_result(z, D)

        x_hat = (MLE_inv @ result_vector)[: net.size * 2]
        print(x_hat)
        print(net.root.state_vector())
        residuals += np.linalg.norm(x_hat - net.root.state_vector())

    print(residuals / 100)
