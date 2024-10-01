# from numpy.typing import NDArray
# import numpy as np
# from p5_simulation.trees import NetworkNode
#
# def create_D_matrix(indices: list[int], total_nodes: int) -> NDArray:
#     D = np.zeros((len(indices),total_nodes*2))
#     for (i, index) in enumerate(indices):
#         D[i,index]=1
#     return D
#
# def all_error_vector(voltage_stdev: float, current_stdev: float, total_nodes: int) -> NDArray:
#     error_v = np.random.normal(size=total_nodes, scale = voltage_stdev)
#     error_i = np.random.normal(size=total_nodes, scale = current_stdev)
#     return np.concatenate((error_v,error_i))
#
# def compute_z_vector(root: NetworkNode, D: NDArray, voltage_stdev: float, current_stdev: float, total_nodes: int) -> NDArray:
#         errors = all_error_vector(voltage_stdev, current_stdev, total_nodes)
#         x = root.state_vector(total_nodes)
#         return D @ (x + errors)
#
# def MLE_matrix(root: NetworkNode, D: NDArray, total_nodes: int) -> NDArray:
#     C = root.all_equations(total_nodes)
#     return np.block([[D.T @ D,C.T],[C,np.zeros((C.shape[0],C.shape[0]))]])
#
# def MLE_result(root: NetworkNode, z: NDArray, D: NDArray, total_nodes: int) -> NDArray:
#     C = root.all_equations(total_nodes)
#     return np.concatenate((D.T@z,np.zeros(C.shape[0])))
