from __future__ import annotations
from typing import Self, Optional
from functools import cached_property
import numpy as np
from numpy.typing import NDArray

SOURCE_VOLTAGE = 240

Resistance = float
Voltage = float
Current = float


# Very particular tree implementation for our needs. B).
class NetworkNode:
    parent: Optional[NetworkNode]
    children: list[tuple[NetworkNode, Resistance]]
    iresistance: Resistance | None

    def __init__(self, network, parent, children, iresistance) -> None:
        self.network = network
        self.parent = parent
        self.children = children
        self.iresistance = iresistance

    @cached_property
    def resistance(self) -> Resistance:
        sum_of_reciprocals = 0
        if self.iresistance is not None:
            sum_of_reciprocals += 1 / self.iresistance
        for child, r in self.children:
            sum_of_reciprocals += 1 / (r + child.resistance)
        return 1 / sum_of_reciprocals

    def get_direct_child_resistance(self, child: NetworkNode) -> Resistance:
        for c, r in self.children:
            if c is child:
                return r
        raise Exception

    @cached_property
    def voltage(self) -> Voltage:
        if self.parent is None:
            return SOURCE_VOLTAGE
        return (
            self.parent.voltage
            - self.current * self.parent.get_direct_child_resistance(self)
        )

    @cached_property
    def current(self) -> Current:
        if self.parent is None:
            return self.voltage / self.resistance
        return self.parent.voltage / (
            self.parent.get_direct_child_resistance(self) + self.resistance
        )

    def add_child(self, child: NetworkNode, resistance: Resistance):
        self.children.append((child, resistance))

    def set_node_indices(self, next_index: int) -> int:
        self.index = next_index
        for child, _ in self.children:
            next_index += 1
            next_index = child.set_node_indices(next_index)
        return next_index

    def voltage_index(self):
        return self.index

    def current_index(self) -> int:
        return self.index + self.network.size

    def equations(self) -> NDArray:
        size = self.network.size
        eqs = []
        if self.parent is not None:
            ohm = np.zeros(size * 2)
            ohm[self.parent.voltage_index()] = 1
            ohm[self.voltage_index()] = -1
            ohm[self.current_index()] = -self.parent.get_direct_child_resistance(self)
            eqs.append(ohm)
        if len(self.children) != 0:
            kirchoff = np.zeros(size * 2)
            kirchoff[self.current_index()] = 1
            for child, _ in self.children:
                kirchoff[child.current_index()] = -1
            eqs.append(kirchoff)
        if len(eqs) == 1:
            return eqs[0]
        else:
            return np.vstack(tuple(eqs))

    def all_equations(self) -> NDArray:
        eqs = self.equations()
        for child, _ in self.children:
            eqs = np.vstack((eqs, child.all_equations()))
        return eqs

    def state_vector(self) -> NDArray:
        vector = np.zeros(self.network.size * 2)
        self.set_state_entries(vector)
        return vector

    def set_state_entries(self, state: NDArray):
        state[self.voltage_index()] = self.voltage
        state[self.current_index()] = self.current
        for child, _ in self.children:
            child.set_state_entries(state)

    def print_stats(self):
        print(f"Index: {self.index}")
        print(f"Voltage: {self.voltage}")
        print(f"Current: {self.current}")
        print(f"Resistance: {self.resistance}")


class Network:
    size: int
    root: NetworkNode
    nodes: [NetworkNode]

    @classmethod
    def singleton(cls) -> Self:
        net = cls()
        root = NetworkNode(net, None, [], None)
        root.index = 0
        net.root = root
        net.nodes = [root]
        net.size = 1

    # Shorthand for network creation
    # each connection is [parent, res] or [parent, res, ires] in the case of a sink
    @classmethod
    def from_connections(cls, cons: list[list[int | float]]) -> Self:
        net = cls()
        root = NetworkNode(net, None, [], None)
        root.index = 0
        nodes = [None] * (len(cons) + 1)
        nodes[0] = root

        for con in cons:
            parent = nodes[con[0]]
            if parent is None:
                raise Exception(f"Parent {con[0]} is none")
            iresistance = None if len(con) < 4 else Resistance(con[3])
            node = NetworkNode(net, parent, [], iresistance)
            node.index = con[1]
            if nodes[con[1]] is not None:
                raise Exception(f"Attempting to write to node {con[1]} twice")
            nodes[con[1]] = node
            parent.add_child(node, Resistance(con[2]))

        net.root = root
        net.nodes = nodes
        net.size = len(nodes)

        return net

    def create_D_matrix(self, indices: list[int]) -> NDArray:
        D = np.zeros((len(indices), self.size * 2))
        for i, index in enumerate(indices):
            D[i, index] = 1
        return D

    def all_error_vector(self, voltage_stdev: float, current_stdev: float) -> NDArray:
        error_v = np.random.normal(size=self.size, scale=voltage_stdev)
        error_i = np.random.normal(size=self.size, scale=current_stdev)
        return np.concatenate((error_v, error_i))

    def compute_z_vector(
        self,
        D: NDArray,
        voltage_stdev: float,
        current_stdev: float,
    ) -> NDArray:
        errors = self.all_error_vector(voltage_stdev, current_stdev)
        x = self.root.state_vector()
        return D @ (x + errors)

    def MLE_matrix(self, D: NDArray) -> NDArray:
        C = self.root.all_equations()
        return np.block([[D.T @ D, C.T], [C, np.zeros((C.shape[0], C.shape[0]))]])

    def MLE_result(self, z: NDArray, D: NDArray) -> NDArray:
        C = self.root.all_equations()
        return np.concatenate((D.T @ z, np.zeros(C.shape[0])))

    def print_node_stats(self):
        for node in self.nodes:
            node.print_stats()
            print()
