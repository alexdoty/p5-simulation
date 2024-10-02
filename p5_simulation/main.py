import numpy as np
from p5_simulation.trees import Network


def main():
    net = Network.from_connections(
        [
            # [0,1, 40, 1000],
            # [0,2, 30, 1000],

            # [0, 1, 40+40j],
            # [0, 2, 50 + 40j, 2_000 + 1_000j],
            # [1, 3, 50 + 60j, 2_000 + 3_000j],
            # [1, 4, 50 + 30j, 2_000 + 2_000j],

            # [0,1,10+10j, 1000+1000j],
            # [0,2,5+10j, 1000+1000j],
            # [0,3,10+10j, 1000+1000j],
            # [2,4,10+20j, 2000+1000j],
            # [0,5,10+10j, 3000+1000j],
            # [0,6,10+10j, 5000+1000j],
            # [4,7,10+30j, 500+1000j],
            # [0,8,10+10j, 1000+4000j],
            # [7,9,10+10j, 1000+2000j],
            # [0,10,40+10j, 1000+1000j],
            # [1,11,10+10j, 1000+9000j],
            # [0,12,10+10j, 1000+1000j],

            [0,1, ]
        ]
    )

    net.print_node_stats()


    D = net.create_D_matrix([0,5,6,7,8,9])

    u_stdev = 1e-3
    i_stdev = 1e-3

    P = D @ np.diag([u_stdev] * net.size + [i_stdev] * net.size) @ D.T

    C= net.root.all_equations()


    G1 = 2 * D.T @ np.linalg.inv(P) @ D

    G_bar = np.block([[G1, np.zeros_like(G1)], [np.zeros_like(G1) , G1]])
    C_bar = np.block([[C, np.zeros_like(C)], [np.zeros_like(C) , C]])

    A = np.block([[G_bar, C_bar.T.conj()], [C_bar, np.zeros([C_bar.shape[0], C_bar.shape[0]])]])
    #A_inv = np.linalg.inv(A)
    #F11 = A_inv[:G_bar.shape[0],:G_bar.shape[1]]

    residuals = 0
    runs = 2
    for _ in range(2):

        z = net.compute_z_vector(D, u_stdev, i_stdev)

        g = 2 * D.T @ np.linalg.inv(P) @ z
        g_bar = np.concatenate([g, g.conj()])

        #x_hat = (F11 @ np.concatenate([g, g.conj()]))[:26]

        #A_small = np.block([[G_bar, C_bar.T.conj()]])
        #print(A_small.shape)
        print(g_bar.shape)
        x_hat = np.linalg.lstsq(A, np.concatenate([g_bar, np.zeros(C_bar.shape[0])]))[0][:net.size * 2]
        print(x_hat.shape)

        print(x_hat)
        print(net.root.state_vector())

        residuals += np.linalg.norm(x_hat - net.root.state_vector())
    print(residuals / runs)


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
