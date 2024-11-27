from __future__ import annotations
import math
from typing import Self, Optional
from functools import cached_property
import numpy as np
from numpy.typing import NDArray
from math import pi, tau
import cmath
import enum
from typing import Any

from p5_simulation import draw
from p5_simulation.utils import pretty, normal_quantile, normal_characteristic, augment_matrices, augment_matrix

SOURCE_VOLTAGE = 325.0 + 0.0j

Resistance = float
Voltage = complex
Current = complex
Impedance = complex

Numeric = int | float | complex


class MeterType(enum.Enum):
    NONE = 1
    EM = 2
    PMU = 3
    VOLTAGE = 4
    CURRENT = 5


# Very particular tree implementation for our needs. B).
class NetworkNode:
    parent: Optional[NetworkNode]
    children: list[tuple[NetworkNode, Impedance]]
    i_impedance: Impedance | None
    meter: MeterType

    measured_phi: Optional[float] = None
    measured_theta: Optional[float] = None
    measured_v: Optional[float] = None
    measured_i: Optional[float] = None

    next_neighbor: Optional[NetworkNode] = None
    prev_neighbor: Optional[NetworkNode] = None
    angle: float = 0
    error: float = 0

    def __init__(self, network, parent, meter, i_impedance) -> None:
        self.network = network
        self.parent = parent
        self.children = []
        self.meter = meter
        self.i_impedance = i_impedance

    # Impedance between the nodes live and neutral
    @cached_property
    def impedance(self) -> Impedance:
        sum_of_reciprocals = 0
        if self.i_impedance is not None:
            sum_of_reciprocals += 1 / self.i_impedance
        for child, imp in self.children:
            sum_of_reciprocals += 1 / (imp + child.impedance)
        return 1 / sum_of_reciprocals

    def get_direct_child_impedance(self, child: NetworkNode) -> Impedance:
        for c, imp in self.children:
            if c is child:
                return imp
        raise Exception

    # Voltage between the nodes live and neutral
    @cached_property
    def voltage(self) -> Voltage:
        if self.parent is None:
            return SOURCE_VOLTAGE
        return (
            self.parent.voltage
            - self.current * self.parent.get_direct_child_impedance(self)
        )

    # Current going from the parent to self
    @cached_property
    def current(self) -> Current:
        if self.parent is None:
            return self.voltage / self.impedance
        return self.parent.voltage / (
            self.parent.get_direct_child_impedance(self) + self.impedance
        )

    def add_child(self, child: NetworkNode, impedance: Impedance):
        self.children.append((child, impedance))

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
            ohm = np.zeros(size * 2, dtype=complex)
            ohm[self.parent.voltage_index()] = 1
            ohm[self.voltage_index()] = -1
            ohm[self.current_index()] = -self.parent.get_direct_child_impedance(self)
            eqs.append(ohm)
        if len(self.children) != 0:
            kirchoff = np.zeros(size * 2, dtype=complex)
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

    def print_stats(self):
        print(f"Index: {self.index}")
        print(f"Voltage: {self.voltage}")
        print(f"Current: {self.current}")
        print(f"Impedance: {self.impedance}")

    def set_neighbors(
        self, last_seen: list[NetworkNode], generation_sizes: list[int], generation: int
    ) -> tuple[list[NetworkNode], list[int]]:
        self.generation = generation
        if self.parent is not None:
            if generation >= len(last_seen):
                last_seen.append(self)
                generation_sizes.append(1)
            else:
                last_seen[generation].next_neighbor = self
                self.prev_neighbor = last_seen[generation]
                last_seen[generation] = self
                generation_sizes[generation] += 1
        else:
            last_seen.append(self)
            generation_sizes.append(1)
        for child, _ in self.children:
            last_seen, generation_sizes = child.set_neighbors(
                last_seen, generation_sizes, generation + 1
            )
        if self.parent is None:
            for node in last_seen:
                cur_node = node
                print(cur_node.index, "aaaa")
                while cur_node.prev_neighbor is not None:
                    cur_node = cur_node.prev_neighbor
                    print(cur_node.index, "bbbb")
                cur_node.prev_neighbor = node
                node.next_neighbor = cur_node

        return (last_seen, generation_sizes)

    def update_angular_error_derivatives(self, generation, generation_sizes):
        if self.parent is None:
            return

        error_derivative = 0.0

        k1 = 0.02
        k2 = 0.1

        if self.parent.parent is not None:
            parent_angle = (self.parent.angle - self.angle) % tau
            if parent_angle < pi:
                parent_angle *= -1
            else:
                parent_angle = tau - parent_angle
            error_derivative += k1 * parent_angle

        for child, _ in self.children:
            parent_angle = (self.angle - child.angle) % tau
            if parent_angle < pi:
                parent_angle *= -1
            else:
                parent_angle = tau - parent_angle
            error_derivative -= k1 * parent_angle

        if self.next_neighbor != self:
            angle_diff = (self.angle - self.next_neighbor.angle) % tau
            error_derivative -= k2 * math.exp(-k2 * angle_diff)

            angle_diff = (self.prev_neighbor.angle - self.angle) % tau
            error_derivative += k2 * math.exp(-k2 * angle_diff)

        self.error_derivative = error_derivative

    def update_angle(self, step_factor: float):
        self.angle -= self.error_derivative * step_factor
        self.angle %= tau
        # print(self.index, self.angle)



s = 1


class Network:
    size: int
    root: NetworkNode
    nodes: list[NetworkNode]

    source_voltage: Voltage = 325.0 + 0.0j
    theta_stdev: float = 0.003 * s
    phi_stdev: float = 0.01 * s
    voltage_rel_err: float = 0.01 * s
    current_rel_err: float = 0.03 * s
    beta: float = 0.99
    constraint_matrix: Optional[NDArray] = None
    metering_matrix: Optional[NDArray] = None

    @classmethod
    def singleton(cls) -> Self:
        net = cls()
        root = NetworkNode(net, None, MeterType.PMU, None)
        root.index = 0
        net.root = root
        net.nodes = [root]
        net.size = 1
        return net

    # Shorthand for network creation
    # each connection is [from, to, meter, imp] or [from, to, meter, imp, i_imp] in the case of a sink
    @classmethod
    def from_connections(cls, cons: list[list[Any]]) -> Self:
        net = cls()
        root = NetworkNode(net, None, MeterType.PMU, None)
        root.index = 0
        nodes = [None] * (len(cons) + 1)
        nodes[0] = root

        for con in cons:
            parent = nodes[con[0]]
            if parent is None:
                raise Exception(f"Parent {con[0]} is none")
            i_impedance = None if len(con) < 5 else Impedance(con[4])
            node = NetworkNode(net, parent, con[2], i_impedance)
            node.index = con[1]
            if nodes[con[1]] is not None:
                raise Exception(f"Attempting to write to node {con[1]} twice")
            nodes[con[1]] = node
            parent.add_child(node, Impedance(con[3]))

        net.root = root
        net.nodes = nodes
        net.size = len(nodes)

        return net

    def state_vector(self) -> NDArray:
        state = np.zeros(self.size * 2, dtype=complex)
        for node in self.nodes:
            state[node.voltage_index()] = node.voltage
            state[node.current_index()] = node.current
        return state

    def create_D_matrix(self) -> NDArray:
        indices = []
        for node in self.nodes:
            match node.meter:
                case MeterType.EM | MeterType.PMU:
                    indices.append(node.voltage_index())
                    indices.append(node.current_index())
                case MeterType.VOLTAGE:
                    indices.append(node.voltage_index())
                case MeterType.CURRENT:
                    indices.append(node.current_index())
        indices = sorted(indices)
        D = np.zeros((len(indices), self.size * 2))
        for i, index in enumerate(indices):
            D[i, index] = 1

        self.metering_matrix = D
        return D

    def create_C_matrix(self) -> NDArray:
        C = self.root.all_equations()
        self.constraint_matrix = C
        return C

    def realize_measurements(self):
        import random

        z = np.zeros(self.size * 2, dtype=complex)

        r_0 = normal_quantile((1 + self.beta) / 2, 1)

        for node in self.nodes:
            voltage = node.voltage
            current = node.current
            v = abs(voltage)
            i = abs(current)
            theta = cmath.phase(voltage)
            phi = cmath.phase(current) - theta

            v_stdev = v * self.voltage_rel_err / r_0
            i_stdev = i * self.current_rel_err / r_0

            v_err = v + random.normalvariate(0.0, v_stdev)
            i_err = i + random.normalvariate(0.0, i_stdev)
            phi_err = phi + random.normalvariate(0.0, self.phi_stdev)

            if node.meter != MeterType.EM:
                theta_err = theta + random.normalvariate(0.0, self.theta_stdev)
                node.measured_theta = theta_err
            else:
                node.measured_theta = node.parent.measured_theta
                theta_err = node.measured_theta
                # node.measured_theta = 0

            voltage_measure = cmath.rect(v_err, theta_err)
            current_measure = cmath.rect(i_err, phi_err + theta_err)

            node.measured_v = v_err
            node.measured_i = i_err
            node.measured_phi = phi_err

            z[node.voltage_index()] = voltage_measure
            z[node.current_index()] = current_measure

        return z

    def compute_sigmas(self):
        voltage_variances = [0.0] * self.size
        voltage_pvariances = [0.0] * self.size
        current_variances = [0.0] * self.size
        current_pvariances = [0.0] * self.size

        r_0 = normal_quantile((1 + self.beta) / 2, 1)

        for k in range(self.size):
            node = self.nodes[k]

            voltage_stdev = node.measured_v * self.voltage_rel_err / r_0
            current_stdev = node.measured_i * self.current_rel_err / r_0

            voltage_variances[k] = (
                1 - normal_characteristic(self.theta_stdev, 1.0) ** 2
            ) * node.measured_v**2 + voltage_stdev**2

            current_variances[k] = (
                1
                - normal_characteristic(self.theta_stdev, 1.0) ** 2
                * normal_characteristic(self.phi_stdev, 1.0) ** 2
            ) * node.measured_i**2 + current_stdev**2

            voltage_pvariances[k] = cmath.exp(2j * node.measured_theta) * (
                (node.measured_v**2 + voltage_stdev**2)
                * normal_characteristic(self.theta_stdev, 2.0)
                - node.measured_v**2 * normal_characteristic(self.theta_stdev, 1.0) ** 2
            )

            current_pvariances[k] = cmath.exp(
                2j * (node.measured_phi + node.measured_theta)
            ) * (
                (node.measured_i**2 + current_stdev**2)
                * normal_characteristic(self.theta_stdev, 2.0)
                * normal_characteristic(self.phi_stdev, 2.0)
                - node.measured_i**2
                * normal_characteristic(self.theta_stdev, 1.0) ** 2
                * normal_characteristic(self.phi_stdev, 1.0) ** 2
            )

        if self.metering_matrix is None:
            D = self.create_D_matrix()
        else:
            D = self.metering_matrix

        sigma_1 = D @ np.diag(voltage_variances + current_variances) @ D.T
        sigma_2 = D @ np.diag(voltage_pvariances + current_pvariances) @ D.T

        return (sigma_1, sigma_2)

    def compute_true_sigmas(self):
        voltage_variances = [0.0] * self.size
        voltage_pvariances = [0.0] * self.size
        current_variances = [0.0] * self.size
        current_pvariances = [0.0] * self.size

        r_0 = normal_quantile((1 + self.beta) / 2, 1)

        for k in range(self.size):
            node = self.nodes[k]

            theta = cmath.phase(node.voltage)
            phi = cmath.phase(node.current) - theta

            voltage_stdev = abs(node.voltage) * self.voltage_rel_err / r_0
            current_stdev = abs(node.current) * self.current_rel_err / r_0

            voltage_variances[k] = (
                1 - normal_characteristic(self.theta_stdev, 1.0) ** 2
            ) * abs(node.voltage)**2 + voltage_stdev**2

            current_variances[k] = (
                1
                - normal_characteristic(self.theta_stdev, 1.0) ** 2
                * normal_characteristic(self.phi_stdev, 1.0) ** 2
            ) * abs(node.current)**2 + current_stdev**2

            voltage_pvariances[k] = cmath.exp(2j * theta) * (
                (abs(node.voltage)**2 + voltage_stdev**2)
                * normal_characteristic(self.theta_stdev, 2.0)
                - abs(node.voltage)**2 * normal_characteristic(self.theta_stdev, 1.0) ** 2
            )

            current_pvariances[k] = cmath.exp(
                2j * (phi + theta)
            ) * (
                (abs(node.current)**2 + current_stdev**2)
                * normal_characteristic(self.theta_stdev, 2.0)
                * normal_characteristic(self.phi_stdev, 2.0)
                - abs(node.current)**2
                * normal_characteristic(self.theta_stdev, 1.0) ** 2
                * normal_characteristic(self.phi_stdev, 1.0) ** 2
            )

        if self.metering_matrix is None:
            D = self.create_D_matrix()
        else:
            D = self.metering_matrix

        sigma_1 = D @ np.diag(voltage_variances + current_variances) @ D.T
        sigma_2 = D @ np.diag(voltage_pvariances + current_pvariances) @ D.T

        return (sigma_1, sigma_2)

    def compute_A_and_g(self, z, sigma_1, sigma_2) -> (NDArray, NDArray):
        if self.metering_matrix is None:
            D = self.create_D_matrix()
        else:
            D = self.metering_matrix

        if self.constraint_matrix is None:
            C = self.create_C_matrix()
        else:
            C = self.constraint_matrix

        B = np.linalg.inv(sigma_1 - sigma_2.conj() @ np.linalg.inv(sigma_1) @ sigma_2)
        W = sigma_2 @ np.linalg.inv(sigma_1)

        G1 = D.T @ B @ D
        G2 = -D.T @ W @ B @ D

        G_bar = augment_matrices(G1, G2)
        C_bar = augment_matrix(C)

        A = np.block(
            [[G_bar, C_bar.T.conj()], [C_bar, np.zeros([C_bar.shape[0], C_bar.shape[0]])]]
        )

        g = D.T @ B @ (z - W @ z.conj())

        return (A, g)

    def compute_A(self, sigma_1, sigma_2) -> NDArray:
        if self.metering_matrix is None:
            D = self.create_D_matrix()
        else:
            D = self.metering_matrix

        if self.constraint_matrix is None:
            C = self.create_C_matrix()
        else:
            C = self.constraint_matrix

        B = np.linalg.inv(sigma_1 - sigma_2.conj() @ np.linalg.inv(sigma_1) @ sigma_2)
        W = sigma_2 @ np.linalg.inv(sigma_1)

        G1 = D.T @ B @ D
        G2 = -D.T @ W @ B @ D

        G_bar = augment_matrices(G1, G2)
        C_bar = augment_matrix(C)

        A = np.block(
            [[G_bar, C_bar.T.conj()], [C_bar, np.zeros([C_bar.shape[0], C_bar.shape[0]])]]
        )
        return A

    def compute_F11_matrix(self, A) -> NDArray:
        A_inv = np.linalg.inv(A)

        return A_inv[:self.size * 4, :self.size*4]

    def print_node_stats(self):
        for node in self.nodes:
            node.print_stats()
            print()

    def set_angles(self):
        last_seen, generation_sizes = self.root.set_neighbors([], [], 0)
        for g, node in enumerate(last_seen):
            node.angle = 0
            cur_node = node.next_neighbor
            i = 0
            while cur_node is not node:
                i += 1
                cur_node.angle = tau / generation_sizes[g] * i
                cur_node = cur_node.next_neighbor

        for node in self.nodes[1:]:
            print(node.index, node.angle)

        for i in range(100_000):
            sum_of_squares = 0
            for node in self.nodes:
                if node.parent is not None:
                    node.update_angular_error_derivatives(1, [])
                    sum_of_squares += node.error_derivative**2

            for node in self.nodes:
                if node.parent is not None:
                    node.update_angle(0.003)

            if sum_of_squares < 0.00001:
                print(i)
                break

        for node in self.nodes[1:]:
            print(node.index, node.angle)

    def draw(self, pos: tuple[int, int] = (0, 0)):
        W = 2560
        H = 1440

        import pygame as pg
        from pygame import gfxdraw

        pg.init()
        pg.font.init()

        layer0: pg.Surface = pg.display.set_mode((W, H))
        layer1: pg.Surface = layer0
        pg.display.set_caption("Network Graph")
        exit: bool = False

        font = pg.font.Font(pg.font.get_default_font(), 30)

        manager = draw.DrawManager(layer0, layer1, font)
        for node in self.nodes:
            x = math.cos(node.angle) * node.generation * 300 + W / 2
            y = math.sin(node.angle) * node.generation * 300 + H / 2
            if node.i_impedance is not None:
                _ = draw.Node.to_manager(
                    manager, ((int(x), int(y)), pretty(node.i_impedance), node.index)
                )
            else:
                _ = draw.Node.to_manager(manager, ((int(x), int(y)), "", node.index))

            if node.parent is not None:
                px = math.cos(node.parent.angle) * node.parent.generation * 300 + W / 2
                py = math.sin(node.parent.angle) * node.parent.generation * 300 + H / 2
                _ = draw.Edge.to_manager(
                    manager,
                    (
                        ((int(px), int(py)), (int(x), int(y))),
                        pretty(node.parent.get_direct_child_impedance(node)),
                        node.index,
                    ),
                )

        manager.setup()

        while not exit:
            layer0.fill(draw.colors["black"])
            manager.draw()
            for event in pg.event.get():
                print(event.type)
                if event.type == pg.QUIT or event.type == 769:
                    exit = True
            pg.display.update()
        pg.quit()
