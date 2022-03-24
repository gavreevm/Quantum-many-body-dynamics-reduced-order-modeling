from jax import numpy as jnp
from jax import vmap
from simulators.mps import mps, mpo, truncate_forward_canonical, truncate_very_last_edge_backward_canonical, set_to_backward_canonical, set_to_forward_canonical, mpo_mps_product
from typing import Union, List, Tuple
from simulators.mps_utils import _set_rank
from simulators.dataclasses import ROMDynamicsGenerator

ReducedOrderModel = List[ROMDynamicsGenerator]

def _layer2lattice(gates_layer: jnp.ndarray, discrete_time: int) -> List[Union[mps, mpo]]:
    """[This function returns lattice tensor network from a layer of unitary gates.]

    Args:
        (complex valued jnp.ndarray of shape (number of qubits-1, 4, 4)):
                [layer of gates describing dynamics of the whole system]
        discrete_time (int): [number of layers in a circuit]

    Returns:
        List[Union[mps, mpo]]: [tensot network as a list of mps and mpo subnetworks]
    """

    # splitting gates into smaller blocks
    gates_layer = gates_layer.reshape((-1, 2, 2, 2, 2))
    gates_layer = gates_layer.transpose((0, 1, 3, 2, 4))
    gates_layer = gates_layer.reshape((-1, 4, 4))
    u, lmbd, vh = jnp.linalg.svd(gates_layer)
    eta = vmap(lambda x: _set_rank(x, 1e-10))(lmbd)
    eta = eta.max()
    u, vh = u[..., :eta] * lmbd[:, jnp.newaxis, :eta], vh[:, :eta]
    u, vh = u.reshape((-1, 2, 2, eta)), vh.reshape((-1, eta, 2, 2))
    u, vh = u.transpose((0, 1, 3, 2)), vh.transpose((0, 2, 1, 3))

    top_ker = u[0]
    bottom_ker = vh[-1]
    mid_kers = []

    for i, (ker_up, ker_down) in enumerate(zip(u[1:], vh[:-1])):
        if i % 2 == 0:
            ker = jnp.tensordot(ker_up, ker_down, axes=1)
            ker = ker.transpose((0, 2, 3, 1))
            mid_kers.append(ker)
        else:
            ker = jnp.tensordot(ker_down, ker_up, axes=1)
            ker = ker.transpose((0, 1, 3, 2))
            mid_kers.append(ker)
    lattice = discrete_time * [[top_ker] + mid_kers + [bottom_ker]]
    lattice = list(map(list, zip(*lattice)))
    return lattice


def _reduce_from_top(lattice: List[mps],
                     init_states: List[jnp.ndarray],
                     number_of_qubits: int,
                     till_number: int,
                     use_full_truncation: bool,
                     truncate_when: int,
                     eps: float) -> None:
    """[This function reduces upper part of a lattice, it acts inplace]

    Args:
        lattice (List[mps]): [lattice]
        init_states (List[jnp.ndarray]): [initial states of a lattice]
        number_of_qubits (int): [number of qubits in a system]
        till_number (int): [the number of a qubit till which one reduces a lattice]
        use_full_truncation (bool): [flag showing whether to use full truncation or not]
        truncate_when (int): [when bond dimension atchieves truncate_when value one performs truncation]
        eps (float): [truncation accuracy]
    """

    if till_number != 0:
        lattice[0][-1] = jnp.tensordot(lattice[0][-1], init_states[0].reshape((2, 1)), axes=1)
        for _ in range(till_number-1):
            lattice[1][-1] = jnp.tensordot(lattice[1][-1], init_states[1].reshape((2, 1)), axes=[[2], [0]]).transpose((0, 1, 3, 2))
            mpo_mps_product(lattice[1], lattice[0])
            lattice.pop(1)
            init_states.pop(1)
            out_dim = lattice[0][0].shape[0]
            if out_dim > truncate_when:
                if use_full_truncation:
                    set_to_backward_canonical(lattice[0])
                    truncate_very_last_edge_backward_canonical(lattice[0], eps)
                    set_to_forward_canonical(lattice[0])
                    truncate_forward_canonical(lattice[0], eps)
                else:
                    set_to_forward_canonical(lattice[0])
                    truncate_forward_canonical(lattice[0], eps)
        if use_full_truncation:
            set_to_backward_canonical(lattice[0])
            truncate_very_last_edge_backward_canonical(lattice[0], eps)
            set_to_forward_canonical(lattice[0])
            truncate_forward_canonical(lattice[0], eps)
        else:
            set_to_forward_canonical(lattice[0])
            truncate_forward_canonical(lattice[0], eps)


def _reduce_from_bottom(lattice: List[mps],
                        init_states: List[jnp.ndarray],
                        number_of_qubits: int,
                        till_number: int,
                        use_full_truncation: bool,
                        truncate_when: int,
                        eps: float) -> None:
    """[This function reduces lower part of a lattice, it acts inplace]

    Args:
        lattice (List[mps]): [lattice]
        init_states (List[jnp.ndarray]): [initial states of a lattice]
        number_of_qubits (int): [number of qubits in a system]
        till_number (int): [the number of a qubit till which one reduces a lattice]
        use_full_truncation (bool): [flag showing whether to use full truncation of not]
        truncate_when (int): [when bond dimension atchieves truncate_when value one performs truncation]
        eps (float): [truncation accuracy]
    """

    number_of_qubits_to_reduce = (number_of_qubits - 1) - till_number
    if number_of_qubits_to_reduce != 0:
        lattice[-1][-1] = jnp.tensordot(lattice[-1][-1], init_states[-1].reshape((2, 1)), axes=1)
        for _ in range(number_of_qubits_to_reduce-1):
            lattice[-2][-1] = jnp.tensordot(lattice[-2][-1], init_states[-2].reshape((2, 1)), axes=[[2], [0]]).transpose((0, 1, 3, 2))
            mpo_mps_product(lattice[-2], lattice[-1], reverse=True)
            lattice.pop(-2)
            init_states.pop(-2)
            out_dim = lattice[-1][0].shape[0]
            if out_dim > truncate_when:
                if use_full_truncation:
                    set_to_backward_canonical(lattice[-1])
                    truncate_very_last_edge_backward_canonical(lattice[-1], eps)
                    set_to_forward_canonical(lattice[-1])
                    truncate_forward_canonical(lattice[-1], eps)
                else:
                    set_to_forward_canonical(lattice[-1])
                    truncate_forward_canonical(lattice[-1], eps)
        if use_full_truncation:
            set_to_backward_canonical(lattice[-1])
            truncate_very_last_edge_backward_canonical(lattice[-1], eps)
            set_to_forward_canonical(lattice[-1])
            truncate_forward_canonical(lattice[-1], eps)
        else:
            set_to_forward_canonical(lattice[-1])
            truncate_forward_canonical(lattice[-1], eps)


def _build(reduced_lattice: List[Union[mps, mpo]]) -> None:
    """[This function set lattice after reduction to a standardized form.]

    Args:
        reduced_lattice (List[Union[mps, mpo]]): [lattice after reduction]
    """

    if len(reduced_lattice) < 3:
        if reduced_lattice[0][-1].shape[-1] == 2:
            for i, ker in enumerate(reduced_lattice[0]):
                reduced_lattice[0][i] = ker[:, jnp.newaxis].transpose((0, 1, 3, 2))
            reduced_lattice.insert(0, len(reduced_lattice[0]) * [jnp.ones((1, 1, 1))])
        else:
            for i, ker in enumerate(reduced_lattice[-1]):
                reduced_lattice[-1][i] = ker[..., jnp.newaxis]
            reduced_lattice.append(len(reduced_lattice[0]) * [jnp.ones((1, 1, 1))])


# TODO: add tests to this function
def _max_bond_dim(reduced_order_model: ReducedOrderModel) -> Tuple[int, int]:
    """This function returns max bond dimension for upper and lower parts of
    reduced order model.

    Args:
        reduced_order_model (ReducedOrderModel): Reduced order model

    Returns:
        Tuple[int, int]: Tuple with max top bond dim. and max bottom bond dim.
    """

    left_bonds_top = [kers.ker_top.shape[0] for kers in reduced_order_model]
    right_bonds_top = [kers.ker_top.shape[2] for kers in reduced_order_model]
    max_top = max(left_bonds_top + right_bonds_top)
    left_bonds_bottom = [kers.ker_bottom.shape[0] for kers in reduced_order_model]
    right_bonds_bottom = [kers.ker_bottom.shape[2] for kers in reduced_order_model]
    max_bottom = max(left_bonds_bottom + right_bonds_bottom)
    return (max_top, max_bottom)
