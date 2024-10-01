from __future__ import annotations
from typing import Any, Self, Optional
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

    def __init__(self, parent, children, iresistance) -> None:
        self.parent = parent
        self.children = children
        self.iresistance = iresistance

    @cached_property
    def resistance(self) -> Resistance:
        sum_of_reciprocals = 0
        if self.iresistance is not None:
            sum_of_reciprocals += 1 / self.iresistance
        for (child, r) in self.children:
           sum_of_reciprocals += 1 / (r + child.resistance)
        return 1 / sum_of_reciprocals

    def get_direct_child_resistance(self, child: NetworkNode) -> Resistance:
        for (c, r) in self.children:
            if c is child:
                return r
        raise Exception

    @cached_property
    def voltage(self) -> Voltage:
        if self.parent is None:
            return SOURCE_VOLTAGE
        return self.parent.voltage - self.current * self.parent.get_direct_child_resistance(self)

    @cached_property
    def current(self) -> Current:
        if self.parent is None:
            return self.voltage / self.resistance
        return self.parent.voltage / (self.parent.get_direct_child_resistance(self) + self.resistance)

    def add_child(self, child: NetworkNode, resistance: Resistance):
        self.children.append((child, resistance))

    def set_node_indices(self, next_index: int) -> int:
        self.index = next_index
        for (child, _) in self.children:
            next_index += 1
            next_index = child.set_node_indices(next_index)
        return next_index

    def voltage_index(self):
        return self.index

    def current_index(self, total_nodes: int) -> int:
        return self.index + total_nodes

    def equations(self, total_nodes: int) -> NDArray:
        eqs = np.zeros(total_nodes * 2)
        eqs[self.voltage_index()] = 1
        eqs[self.current_index(total_nodes)] = -self.resistance
        if len(self.children) != 0:
            kirchoff = np.zeros(total_nodes * 2)
            kirchoff[self.current_index(total_nodes)] = 1
            for (child, _) in self.children:
                kirchoff[child.current_index(total_nodes)] = -1
            eqs = np.vstack((eqs, kirchoff))
        return eqs

    def all_equations(self, total_nodes: int) -> NDArray:
        eqs = self.equations(total_nodes)
        for (child, _) in self.children:
            eqs = np.vstack((eqs, child.all_equations(total_nodes)))
        return eqs

    def state_vector(self, total_nodes: int) -> NDArray:
        vector = np.zeros(total_nodes * 2)
        self.set_state_entries(vector, total_nodes)
        return vector

    def set_state_entries(self, state: NDArray, total_nodes: int):
        state[self.voltage_index()] = self.voltage
        state[self.current_index(total_nodes)] = self.current
        for (child, _) in self.children:
            child.set_state_entries(state, total_nodes)

    def print_stats(self):
        print(f"Voltage: {self.voltage}")
        print(f"Current: {self.current}")
        print(f"Resistance: {self.resistance}")
