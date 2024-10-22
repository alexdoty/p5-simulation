from __future__ import annotations
import math
from typing import Self, Optional
from functools import cached_property
import numpy as np
from numpy import typing
from numpy.typing import NDArray
from math import pi, tau
import cmath
import enum


from p5_simulation import draw
from p5_simulation.utils import pretty

SOURCE_VOLTAGE = 240.0 + 0.0j

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

    next_neighbor: Optional[NetworkNode] = None
    prev_neighbor: Optional[NetworkNode] = None
    angle = 0
    error = 0

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

    def update_angular_error_derivatives(
        self, generation, generation_sizes
    ):
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
        #print(self.index, self.angle)




from typing import Any

class Network:
    size: int
    root: NetworkNode
    nodes: list[NetworkNode]

    source_voltage: Voltage = 240.0 + 0.0j
    phase_stdev: float = 0.1
    voltage_stdev: float = 0.1
    current_stdev: float = 0.1

    @classmethod
    def singleton(cls) -> Self:
        net = cls()
        root = NetworkNode(net, None, None)
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

        D = np.zeros((len(indices), self.size * 2))
        for i, index in enumerate(indices):
            D[i, index] = 1
        return D

    def realize_measurements(self):
        import random
        z = np.zeros(self.size * 2, dtype=complex)
        for node in self.nodes:
            voltage = node.voltage
            current = node.current
            v = abs(voltage)
            i = abs(current)
            theta = cmath.phase(voltage)
            phi = cmath.phase(current) - theta

            v_err = v + random.normalvariate(0.0, 3.0)
            i_err = i + random.normalvariate(0.0, 0.02)
            theta_err = theta + random.normalvariate(0.0, 0.003)
            phi_err = phi + random.normalvariate(0.0, 0.01)

            if node.meter == MeterType.PMU:
                node.theta = theta_err
            else:
                node.theta = node.parent.theta
                theta_err = node.theta
            voltage_measure = cmath.rect(v_err, theta_err)
            current_measure = cmath.rect(i_err, phi_err + theta_err)

            z[node.voltage_index()] = voltage_measure
            z[node.current_index()] = current_measure


        return z

    def all_error_vector(self, voltage_stdev: float, current_stdev: float) -> NDArray:
        error_v = (
            np.random.normal(size=(self.size, 2), scale=voltage_stdev)
            .view(np.complex128)
            .reshape(self.size)
        )
        error_i = (
            np.random.normal(size=(self.size, 2), scale=current_stdev)
            .view(np.complex128)
            .reshape(self.size)
        )
        return np.concatenate((error_v, error_i))

    def compute_z_vector(
        self,
        D: NDArray,
        voltage_stdev: float,
        current_stdev: float,
    ) -> NDArray:
        errors = self.all_error_vector(voltage_stdev, current_stdev)
        x = self.state_vector()
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
                    sum_of_squares += node.error_derivative ** 2

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
                _ = draw.Node.to_manager(manager, ((int(x), int(y)), pretty(node.i_impedance), node.index))
            else:
                _ = draw.Node.to_manager(manager, ((int(x), int(y)), "", node.index))


            if node.parent is not None:
                px = math.cos(node.parent.angle) * node.parent.generation * 300 + W / 2
                py = math.sin(node.parent.angle) * node.parent.generation * 300 + H / 2
                _ = draw.Edge.to_manager(manager, (((int(px), int(py)), (int(x), int(y))), pretty(node.parent.get_direct_child_impedance(node)), node.index))


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
