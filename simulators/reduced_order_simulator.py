from jax import numpy as jnp

from simulators.reduced_order_simulator_utils import _layer2lattice, _reduce_from_top, _reduce_from_bottom, _build
from typing import List

from functools import reduce


ReducedOrderModel = List[jnp.ndarray]


class ReducedOrderSimulator:

    def __init__(self):
        """[Class wrapping reduced-order quantum dynamics simulator.]
        """
        pass

    def build_reduced_order_model(self,
                                  system_qubit_number: int,
                                  controlled_qubit_number: int,
                                  discrete_time: int,
                                  initial_environment_state: List[jnp.ndarray],
                                  gates_layer: jnp.ndarray,
                                  use_full_truncation: bool = True,
                                  truncate_when: int = 512,
                                  eps: float = 1e-5) -> ReducedOrderModel:
        """[This method builds the reduced-order model of quantum dynamics.]

        Args:
            system_qubit_number (int): [number of a qubit that is seen as the target system]
            controlled_qubit_number (int): [number of qubit under control]
            discrete_time (int): [number of time steps]
            initial_environment_state (List[jnp.ndarray]): [states of all qubits of the environemtn]
            gates_layer (complex valued jnp.ndarray of shape (number of qubits-1, 4, 4)):
                [layer of gates describing dynamics of the whole system]
            use_full_truncation (bool): [flag showing whether to use full truncation of not]
            truncate_when (int): [when bond dimension atchieves truncate_when value one performs truncation]
            eps (float): [truncation accuracy]

        Returns:
            ReducedOrderModel: [reduced order model of quantum dynamics]
        """

        assert system_qubit_number == controlled_qubit_number, "Different controlled qubit and system qubit are not supported yet"

        lattice = _layer2lattice(gates_layer, discrete_time)
        number_of_qubits = len(lattice)
        if system_qubit_number == 0:
            _reduce_from_bottom(lattice,
                                initial_environment_state,
                                number_of_qubits,
                                system_qubit_number,
                                use_full_truncation=use_full_truncation,
                                truncate_when=truncate_when,
                                eps=eps)
            _build(lattice)

            
        elif system_qubit_number == (number_of_qubits-1):
            _reduce_from_top(lattice,
                             initial_environment_state,
                             number_of_qubits,
                             system_qubit_number,
                             use_full_truncation=use_full_truncation,
                             truncate_when=truncate_when,
                             eps=eps)
            _build(lattice)
        else:
            _reduce_from_bottom(lattice,
                                initial_environment_state,
                                number_of_qubits,
                                system_qubit_number,
                                use_full_truncation=use_full_truncation,
                                truncate_when=truncate_when,
                                eps=eps)
            _reduce_from_top(lattice,
                             initial_environment_state,
                             number_of_qubits,
                             system_qubit_number,
                             use_full_truncation=use_full_truncation,
                             truncate_when=truncate_when,
                             eps=eps)
            _build(lattice)
        return lattice[0]

    def compute_dynamics(self,
                         reduced_order_model: ReducedOrderModel,
                         control_gates: jnp.ndarray,
                         init_state: jnp.ndarray) -> jnp.ndarray:
        """[This method runs simulation of a system dynamics within reduced-order model]

        Args:
            reduced_order_model (ReducedOrderModel): [reduced-order model]
            control_gates (conplex valued jnp.ndarray of shape (discrete_time, 2, 2)): [description]
            init_state (complex valued jnp.ndarray of shape (2,)): [initial state of a system]

        Returns:
            complex valued jnp.ndarray of shape (discrete_time, 2, 2): [dynamics of density matrices]
        """

        def iter(carry, control_and_gate):
            state, rhos = carry
            gate, control = control_and_gate
            state = jnp.tensordot(gate, state, axes=3)
            state = jnp.tensordot(control, state, axes=[[1], [1]])
            state = state.transpose((1, 0, 2))
            rho = jnp.tensordot(state, state.conj(), axes=[[0, 2], [0, 2]])
            rho = rho / jnp.trace(rho)
            rhos = rhos + [rho[jnp.newaxis]]
            return state, rhos

        _, rhos = reduce(iter, zip(reversed(reduced_order_model), control_gates), (init_state.reshape((1, 2, 1)), []))
        return jnp.concatenate(rhos, axis=0)
