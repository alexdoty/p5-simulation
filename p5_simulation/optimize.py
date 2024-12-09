
from numpy.typing import NDArray
import numpy as np
from random import choice, randint, random
from p5_simulation.utils import augment_matrices, augment_matrix, augment_transformation
from p5_simulation.trees import MeterType, Network, NetworkNode
from queue import Queue
from copy import deepcopy

# Returns max distance depending on temperature
def smoothing_function(x: float, smooth: float, max: int) -> float:
    return abs((max * 1/(np.pi/2) * np.atan(1/smooth * x)))

def setup(net: Network, k: int) -> list[int]:
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

def compare_configurations(c1: int, c2: int) -> bool:
    if abs(c1 - c2) < 0.25:
        return True
    return False

def compute_projmat_and_leverages(net: Network) -> tuple[NDArray, list[float], bool]:
    D = net.create_D_matrix()
    S1, S2 = net.compute_true_sigmas()

    # Compute relevant matrices
    A = net.compute_A(S1, S2)
    if np.linalg.det(A) == 0:
        return A, [], False
    F11 = net.compute_F11_matrix(A)
    B = np.linalg.inv(S1 - S2.conjugate() @ np.linalg.inv(S1) @ S2)
    W = S2 @ np.linalg.inv(S1.conjugate())
    J_aug = augment_matrices(
       D.T @ B,
       -D.T @ B @ W
    )
    D_aug = augment_matrix(D)
    H = D_aug @ F11 @ J_aug

    T = augment_transformation(D.shape[0])
    H_real = np.linalg.inv(T) @ H @ T

    s = (H_real.shape[1] // 4)
    leverages: list[float] = [0] * s
    for i in range(0, s):
        leverages[i] = H_real[i, i] + H_real[i + s, i + s] + H_real[i + 2 * s, i + 2 * s] + H_real[i + 3 * s, i + 3 * s]

    return H_real, leverages, True

def compute_F11(net: Network) -> NDArray:
    D = net.create_D_matrix()
    S1, S2 = net.compute_true_sigmas()
    A = net.compute_A(S1, S2)
    F11 = net.compute_F11_matrix(A)
    return F11

def assessment_metric(old: Network, new: Network) -> float:
    old_F11 = compute_F11(old)
    print("Old F11 is good!")
    new_F11 = compute_F11(new)
    value = 0
    return 1/(4 * new.size) * np.linalg.trace(new_F11 / old_F11)

def anneeling_solve(net: Network, k: int, t: int = 20) -> tuple[Network, list[int]]:
    temp = t
    initial_locations = set(setup(net, k))
    locations = deepcopy(initial_locations)
    dec: float = t/(100 * k)
    level: int = 1
    possible_location: int = 0

    # i = 0
    while temp > 0:
        # i += 1
        temp -= dec
        # _, leverages = compute_projmat_and_leverages(net)
        r = random()
        r2 = random()
        bound = t/level/2

        net.set_meters_anneeling(locations)
        if r <= 0.95 and level > 2:
            possible_location = choice(list(locations))
            not_metered_location = choice(list(initial_locations.difference(locations)))
            union_locations = locations.union({not_metered_location})
            net.set_meters_anneeling(union_locations)
            _, leverages, inv = compute_projmat_and_leverages(net)
            sorted_locs = sorted(list(union_locations))
            if leverages[sorted_locs.index(possible_location)] < leverages[sorted_locs.index(not_metered_location)]:
                if temp > bound:
                    r2 += 0.2
                if r2 > 0.5:
                    locations.add(not_metered_location)
                    locations.remove(possible_location)
        else:
            if level == k+1:
                continue
            _, leverages, inv = compute_projmat_and_leverages(net)
            loc = 0
            lowest = leverages[0]
            for i in range(1, len(leverages)):
                if leverages[i] < lowest:
                    loc = i
                    lowest = leverages[i]
            sorted_locs = sorted(list(locations))
            possible_location = sorted_locs[loc]
            locations.remove(possible_location)
            level += 1

    return net, list(locations)

def test_solve(net: Network, k: int, t: int = 20) -> list[int]:
    # Controls the probability of moving far away from good states.
    temp = t
    # Initializes measuring locations to all locations.
    locations: list[int] = setup(net, k)
    meters: dict[int, MeterType] = {n: MeterType.PMU for n in locations}
    # Keeps track of currently good states. Pop off the back and push to the front
    # when good one is found.
    states = Queue(100)
    state: list[int] = list()
    curnode = net.root
    # How deep we are
    level = 1
    # Use index to get the nodes index, also corresponds to
    # diagonal entry in projection matrix.
    while temp > 0:
        # We want to search through the possible configurations when removing
        # 0 to k measuring stations. At a state, we randomly jump to another one.
        # We save the largest values, and as the temperature falls, the difference
        # between the checked states and the good states fall. Chossing some
        # max number n, we ensure we only have n possible states, so when
        # adding a new state, the previous "worst" state is removed.

        # While the states queue is not full, we pick a random configuration.
        # with k measuring devices removed. When checking a new state, we first
        # run the temperature through the smoothing function to get our max distance d.
        # We then generate a random int between 0 and d. We then pick a configuration that is
        # within distance d from any one of the good configurations on the same level, which
        # is determined by another randomly generated number.

        # Another idea. We start at the top (no removed measuring devices), and move downwards.
        # As the temperature falls, we will start preferring to move a level down. While on a level,
        # we have two possible moves: switch a measuring device for a another one or move a level down.
        # When switching a measuring device, we go a level up and move to a random child node. At the child
        # node, we compare leverages. If its smaller or slightly worse, push to deque on the right (and pop).
        # When moving down, we pick a random configuration that includes the previous removed.

        # Ensure the number of removed PMUs is the same as the level
        if len(state) > level:
            raise ValueError("State has too many elements!")

        r = 1/k * random()
        temp = smoothing_function(temp - r, 10, k)
        if level <= k and temp < t/(level**2):
            level += 1
            continue

        proj, leverages, val = compute_projmat_and_leverages(net)

    return locations

def greedy_solve(net: Network, k: int) -> tuple[Network, list[int]]:
    # Setup the network and the connections
    locations = setup(net, k)

    # Compute H, multiply by T^(-1) on the left and T on the right and get the diagonal entries
    # Find the index with the smallest value
    # Remove that measurement and calculate again
    num_locs = net.size
    for i in range(k):
    #     D = net.create_D_matrix()
    #     S1, S2 = net.compute_true_sigmas()

    #     # Compute relevant matrices
    #     A = net.compute_A(S1, S2)
    #     F11 = net.compute_F11_matrix(A)
    #     B = np.linalg.inv(S1 - S2.conjugate() @ np.linalg.inv(S1) @ S2)
    #     W = S2 @ np.linalg.inv(S1.conjugate())
    #     J_aug = augment_matrices(
    #        D.T @ B,
    #        -D.T @ B @ W
    #     )
    #     D_aug = augment_matrix(D)
    #     H = D_aug @ F11 @ J_aug

    #     T = augment_transformation(D.shape[0])
    #     H_real = np.linalg.inv(T) @ H @ T

    #     # Get arg min of leverages
    #     loc = 0
    #     lowest = H_real[0,0] + H_real[num_locs,num_locs]
    #     for j in range(1, H_real.shape[1] // 2):
    #         if H_real[j, j] + H_real[j+num_locs, j+num_locs] < lowest:
    #             loc = j

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
            print("Broken algorithm ig")
            raise ValueError
        loc = locations.pop(loc)

        # Check if removing voltage meter or current meter
        net.set_meters_anneeling(locations)
        # if loc < net.size:
        #     net.nodes[loc].meter = MeterType.NONE
        # elif loc >= net.size:
        #     loc -= net.size
        #     net.nodes[loc].meter = MeterType.NONE
        num_locs -= 1

    print(locations)
    return net, locations

def greedy_solve_old(net: Network, k: int) -> tuple[Network, list[int]]:
    # Setup the network and the connections
    locations = setup(net, k)

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
            print("Broken algorithm ig")
            raise ValueError
        loc = locations.pop(loc)

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
