""" Reduced order simulator module
"""

from jax import numpy as jnp
from jax import lax, vmap
from functools import reduce
from typing import List, Union

from simulators.reduced_order_simulator_utils import (
    _layer2lattice,
    _reduce_from_top,
    _reduce_from_bottom,
    _build,
    _max_bond_dim,
    ReducedOrderModel)

from simulators.exact_simulator_utils import M_inv, complete_system
from simulators.dataclasses import ROMDynamicsGenerator

PreprocessedReducedOrderModel = ROMDynamicsGenerator  # vectorized one


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
            truncate_when (int): [when bond dimension atchieves truncate_when value one performs
                                                                                     truncation]
            eps (float): [truncation accuracy]

        Returns:
            ReducedOrderModel: [reduced order model of quantum dynamics]
        """

        assert system_qubit_number == controlled_qubit_number, \
        "Different controlled qubit and system qubit are not supported yet"

        lattice = _layer2lattice(gates_layer, discrete_time)
        number_of_qubits = len(lattice)

        assert number_of_qubits % 2 != 0, \
        "Reduced order simulator supports only odd number of qubits"

        isometries = [[], []] # top/bottom isometries
        if system_qubit_number == 0:
            isometries[1] = _reduce_from_bottom(lattice,
                                initial_environment_state,
                                number_of_qubits,
                                system_qubit_number,
                                use_full_truncation=use_full_truncation,
                                truncate_when=truncate_when,
                                eps=eps)
            _build(lattice)


        elif system_qubit_number == (number_of_qubits-1):
            isometries[0] = _reduce_from_top(lattice,
                             initial_environment_state,
                             number_of_qubits,
                             system_qubit_number,
                             use_full_truncation=use_full_truncation,
                             truncate_when=truncate_when,
                             eps=eps)
            _build(lattice)

        else:
            isometries[1] = _reduce_from_bottom(lattice,
                                initial_environment_state,
                                number_of_qubits,
                                system_qubit_number,
                                use_full_truncation=use_full_truncation,
                                truncate_when=truncate_when,
                                eps=eps)

            isometries[0] = _reduce_from_top(lattice,
                             initial_environment_state,
                             number_of_qubits,
                             system_qubit_number,
                             use_full_truncation=use_full_truncation,
                             truncate_when=truncate_when,
                             eps=eps)
            _build(lattice)

        return [ROMDynamicsGenerator(ker_top=ker_top, ker_mid=ker_mid,
                                     ker_bottom=ker_bottom) for ker_top, ker_mid, ker_bottom in zip(
                                                            *lattice)], isometries, lattice


    #TODO: write tests for this method
    def preprocess_reduced_order_model(self,
                                       reduced_order_model:
                                       ReducedOrderModel) -> PreprocessedReducedOrderModel:
        """[This method preprocess reduced-order model to make it suitable for fast jit.]
        Args:
            reduced_order_model (ReducedOrderModel): [reduced-order model]
        Returns:
            PreprocessedReducedOrderModel: [preprocessed reduced-order model]
        """

        # top_left_max_dim, _, top_right_max_dim = reduced_order_model[0].ker_top.shape
        # bottom_left_max_dim, _, bottom_right_max_dim = reduced_order_model[0].ker_bottom.shape
        top_max_dim, bottom_max_dim = _max_bond_dim(reduced_order_model)
        top_tensors = []
        mid_tensors = []
        bottom_tensors = []
        for rom_ker in reduced_order_model:
            ker_top, ker_mid, ker_bottom = rom_ker.ker_top, rom_ker.ker_mid, rom_ker.ker_bottom
            top_left_dim, _, top_right_dim = ker_top.shape
            bottom_left_dim, _, bottom_right_dim = ker_bottom.shape

            top_tensors.append(jnp.pad(ker_top, ((0, top_max_dim - top_left_dim), (0, 0), (
                                                  0, top_max_dim - top_right_dim)))[jnp.newaxis])
            mid_tensors.append(ker_mid[jnp.newaxis])
            bottom_tensors.append(jnp.pad(ker_bottom, ((
                0, bottom_max_dim - bottom_left_dim), (0, 0), (
                0, bottom_max_dim - bottom_right_dim)))[jnp.newaxis])

        ker_top, ker_mid, ker_bottom = jnp.concatenate(
                 top_tensors, axis=0), jnp.concatenate(
                 mid_tensors, axis=0), jnp.concatenate(
                 bottom_tensors, axis=0)

        return ROMDynamicsGenerator(
               ker_top=ker_top, ker_mid=ker_mid, ker_bottom=ker_bottom)

    # TODO: tests for the fast_jit == True
    def compute_dynamics(self,
                         reduced_order_model: Union[ReducedOrderModel,
                                                    PreprocessedReducedOrderModel],
                         control_gates: jnp.ndarray,
                         init_state: jnp.ndarray,
                         fast_jit: bool = False) -> jnp.ndarray:
        """[This method runs simulation of a system dynamics within reduced-order model]

        Args:
            reduced_order_model (Union[ReducedOrderModel, PreprocessedReducedOrderModel]):
                [reduced-order model]
            control_gates (conplex valued jnp.ndarray of shape (discrete_time, 2, 2)): [description]
            init_state (complex valued jnp.ndarray of shape (2,)): [initial state of a system]
            fast_jit (bool) [flag showing whether to use preprocessed reduced-order model for fast
                jit or not]. Defaults to False.

        Returns:
            complex valued jnp.ndarray of shape (discrete_time, 2, 2): [dynamics of density matrices]
        """

        if fast_jit:
            _, top_dim, _, _ = reduced_order_model.ker_top.shape
            _, bottom_dim, _, _ = reduced_order_model.ker_bottom.shape
            init_state = init_state[jnp.newaxis, :, jnp.newaxis]
            init_state = jnp.pad(init_state, ((0, top_dim-1), (0, 0), (0, bottom_dim-1)))

            def iter(state, control_and_gate):
                rom, control = control_and_gate
                top, mid, bottom = rom.ker_top, rom.ker_mid, rom.ker_bottom
                state = jnp.tensordot(top, state, axes=1)
                state = jnp.tensordot(mid, state, axes=[[1, 2], [1, 2]])
                state = jnp.tensordot(bottom, state, axes=[[1, 2], [1, 3]])
                state = jnp.tensordot(control, state, axes=[[1], [1]])
                state = state.transpose((2, 0, 1))
                state /= jnp.linalg.norm(state)
                rho = jnp.tensordot(state, state.conj(), axes=[[0, 2], [0, 2]])
                return state, rho

            _, rhos = lax.scan(iter, init_state, (reduced_order_model, control_gates[::-1]),
                                                                              reverse=True)
            return rhos[::-1]

        else:
            def iter(carry, control_and_gate):
                state, rhos = carry
                rom, control = control_and_gate
                top, mid, bottom = rom.ker_top, rom.ker_mid, rom.ker_bottom
                state = jnp.tensordot(top, state, axes=1)
                state = jnp.tensordot(mid, state, axes=[[1, 2], [1, 2]])
                state = jnp.tensordot(bottom, state, axes=[[1, 2], [1, 3]])
                state = jnp.tensordot(control, state, axes=[[1], [1]])
                state = state.transpose((2, 0, 1))
                state /= jnp.linalg.norm(state)
                rho = jnp.tensordot(state, state.conj(), axes=[[0, 2], [0, 2]])
                rhos = rhos + [rho[jnp.newaxis]]
                return state, rhos

            final_state, rhos = reduce(iter, zip(reversed(reduced_order_model), control_gates), (
                                                init_state.reshape((1, 2, 1)), []))
            return jnp.concatenate(rhos, axis=0), final_state

    # TODO: tests for this method
    def compute_quantum_channels(self,
                                 reduced_order_model: ReducedOrderModel,
                                 control_gates: jnp.ndarray,
                                 fast_jit: bool = False) -> jnp.ndarray:
        """[This method computes quantum channels within reduced-order model]

        Args:
            reduced_order_model (Union[ReducedOrderModel, PreprocessedReducedOrderModel]):
                [reduced-order model]
            control_gates (conplex valued jnp.ndarray of shape (discrete_time, 2, 2)): [description]
            fast_jit (bool) [flag showing whether to use preprocessed reduced-order model for fast
                jit or not]. Defaults to False.

        Returns:
            complex valued jnp.ndarray of shape (discrete_time, 2, 2, 2, 2): [quantum channels]
        """

        fun = vmap(self.compute_dynamics, in_axes=(None, None, 0, None), out_axes=-1)
        phi = fun(reduced_order_model, control_gates, complete_system, fast_jit)
        phi = jnp.tensordot(phi, M_inv, axes=1)
        return phi
