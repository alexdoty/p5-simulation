from math import comb, exp
import sys

from numpy.ma.core import choose
from numpy.typing import NDArray
import numpy as np
from random import choice, randint, random
from p5_simulation.utils import augment_matrices, augment_matrix, augment_transformation
from p5_simulation.trees import MeterType, Network, NetworkNode
from copy import deepcopy
from sortedcontainers import sortedset, SortedSet

EPS = 2e-2

def acceptance_probability(a, b, temp):
    """
    Returns the probability we accept given two leverages a and b and the current temperature
    """
    if b > a:
        return 1
    if abs(val := (a - b) / temp) > 400:
        return 0
    return exp(-val)


def setup(net: Network, k: int) -> list[int]:
    """
    Sets up the network such that each meter is a PMU and returns a list of meter locations
    """
    assert k <= net.size, "k must be less than or equal to the sum of nodes and edges!"
    locations = list(range(0, net.size))

    # Ensure each node has a PMU
    for node in net.nodes:
        node.meter = MeterType.PMU

    return locations


def remove_meter(net: Network, loc: int):
    size = net.size
    if loc < size:
        net.nodes[loc].meter = MeterType.NONE
    else:
        loc -= size
        net.nodes[loc].meter = MeterType.NONE

def compute_projmat_and_leverages(net: Network) -> tuple[NDArray, list[float], bool]:
    """
    Computes the projection matrix H^~ and the leverages as a list.
    """
    inv = True
    D = net.create_D_matrix()
    S1, S2 = net.compute_true_sigmas()

    # Compute relevant matrices
    A = net.compute_A(S1, S2)

    if np.linalg.det(A) == 0:
        inv = False

    F11 = net.compute_F11_matrix(A)
    B = np.linalg.inv(S1 - S2.conjugate() @ np.linalg.inv(S1) @ S2)
    W = S2 @ np.linalg.inv(S1.conjugate())
    J_aug = augment_matrices(D.T @ B, -D.T @ B @ W)
    D_aug = augment_matrix(D)
    H = D_aug @ F11 @ J_aug

    T = augment_transformation(D.shape[0])
    H_real = np.linalg.inv(T) @ H @ T

    s = H_real.shape[1] // 4
    leverages: list[float] = [0] * s
    for i in range(0, s):
        leverages[i] = (
            2
            if max(
                H_real[i, i],
                H_real[i + 2 * s, i + 2 * s],
                H_real[i + s, i + s],
                H_real[i + 3 * s, i + 3 * s],
            ).real
            >= 1 - EPS
            else (
                H_real[i, i]
                + H_real[i + 2 * s, i + 2 * s]
                + H_real[i + s, i + s]
                + H_real[i + 3 * s, i + 3 * s]
            ).real
            / 2
        )

    return H_real, leverages, inv


def compute_F11(net: Network) -> NDArray:
    """
    Computes the F11 matrix.
    """
    D = net.create_D_matrix()
    S1, S2 = net.compute_true_sigmas()
    A = net.compute_A(S1, S2)
    F11 = net.compute_F11_matrix(A)
    return F11


def assessment_metric(old: Network, new: Network) -> float:
    """
    Assessment metric used to compare how (relatively) well the
    PMU placements for the old and new network and estimate the grid-state.
    =1 => old and new networks are equally good
    <1 => new is better than old
    >1 => old is better than new
    """
    old_F11 = compute_F11(old)
    new_F11 = compute_F11(new)
    return 1 / (4 * new.size) * np.linalg.trace(new_F11 / old_F11).real


def test_annealing(net: Network, k: int) -> tuple[Network, list[int]]:
    """
    Alternative implementation of the annealing solver, sometimes works better
    """
    measurables = SortedSet()
    for node in net.nodes:
        if node.meter == MeterType.PMU:
            measurables.add(node.index)

    net, positions = greedy_solve(net, k)
    measured = SortedSet(positions)
    temp = 1
    i = 1
    while temp > 0:
        if i % 100 == 0:
            print(f"iteration: {i}")
        i += 1

        unmeasured = measurables - measured
        node_a = choice(measured)
        node_b = choice(unmeasured)

        r = random()

        net.nodes[node_b].meter = MeterType.PMU
        measured.add(node_b)

        _, leverages, inv = compute_projmat_and_leverages(net)

        if (
            acceptance_probability(
                leverages[measured.index(node_a)],
                leverages[measured.index(node_b)],
                temp,
            )
            > r
            and leverages[measured.index(node_a)].real < 2
        ):
            measured.remove(node_a)
            net.nodes[node_a].meter = MeterType.NONE
        else:
            measured.remove(node_b)
            net.nodes[node_b].meter = MeterType.NONE
        temp -= 5e-4
    return net, list(measured)


def anneeling_solve(net: Network, k: int) -> tuple[Network, list[int]]:
    """
    Attempts to find the best configuration of PMU meter locations given
    k meters are removed using an implementation simulated annealing.
    """
    temp: float = 1
    initial_locations: set[int] = set(setup(net, k))
    locations: set[int] = deepcopy(initial_locations)
    level: int = 1
    possible_location: int = 0

    bincoef: int = comb(len(initial_locations), level)
    dt: float = -1 / (np.log(bincoef) * 100)

    while temp > 0 or level < k + 1:
        temp += dt
        r: float = random()
        r2: float = random()

        # If temperature is high, we ensure swapping
        if temp >= 0.5:
            r = 0

        # Ensure meters are set correctly
        net.set_meters_no_type(locations)

        # Attempt to swap PMUs
        if r <= 0.95 and level > 2:
            # Choose a location that is metered
            possible_location = choice(list(locations))
            # Choose a location that is not metered
            not_metered_location = choice(list(initial_locations.difference(locations)))
            # Create the set that contains the locations and the non-metered location
            union_locations = locations.union({not_metered_location})
            # Sort locations to ensure correct indices
            sorted_locs = sorted(list(union_locations))
            # Set locations based on
            net.set_meters_no_type(union_locations)

            # Compute leverages
            _, leverages, _ = compute_projmat_and_leverages(net)

            # If the leverage is too high, skip (Conjecture 1)
            if leverages[sorted_locs.index(possible_location)] == 2:
                continue

            # Configuration is worse
            if (
                (pl := leverages[sorted_locs.index(possible_location)])
                > (nl := leverages[sorted_locs.index(not_metered_location)])
            ):
                # If the temperature is high, increase probability
                if temp > 0.5:
                    r2 -= 0.1
                # Swap PMU locations
                if r2 < acceptance_probability(pl, nl, temp):
                    locations.add(not_metered_location)
                    locations.remove(possible_location)
            # Configuration is better, so swap PMU locations
            else:
                locations.add(not_metered_location)
                locations.remove(possible_location)

        # Run greedy algorithm for next level
        else:
            # If we are at the desired number of removed meters, skip
            if level == k + 1:
                continue

            # Compute leverages
            _, leverages, _ = compute_projmat_and_leverages(net)

            # Find lowest leverage
            loc = 0
            lowest = leverages[0]
            for i in range(1, len(leverages)):
                if leverages[i] < lowest:
                    loc = i
                    lowest = leverages[i]

            # Sort locations
            sorted_locs = sorted(list(locations))
            # Get the location with lowest leverage
            possible_location = sorted_locs[loc]
            # Remove this location
            locations.remove(possible_location)
            # Increment level
            level += 1

            # Update bincoef and dt
            bincoef = comb(len(initial_locations), level)
            dt = -1 / (np.log(bincoef) * 100)
            temp = 1

    # Ensure meters are set correctly
    net.set_meters_no_type(locations)
    return net, sorted(list(locations))

def greedy_solve(net: Network, k: int) -> tuple[Network, list[int]]:
    """
    Implementation of the greedy solver shown in "Selection of measurements for state estimation in
    electrical power distribution grids".
    """
    # Setup the network and the connections
    locations = setup(net, k)

    # Compute H, multiply by T^(-1) on the left and T on the right and get the diagonal entries
    # Find the index with the smallest value
    # Remove that measurement and calculate again
    num_locs = net.size
    for i in range(k):
        H_real, leverages, inv = compute_projmat_and_leverages(net)
        loc = 0
        lowest = leverages[0]
        for i in range(1, len(leverages)):
            if leverages[i] < lowest:
                loc = i
                lowest = leverages[i]

        if loc >= num_locs:
            print(H_real.shape)
            print(locations)
            raise ValueError
        loc = locations.pop(loc)

        net.set_meters_no_type(locations)
        num_locs -= 1

    return net, locations
