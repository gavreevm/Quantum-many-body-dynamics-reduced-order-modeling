from jax import numpy as jnp
from jax import jit, vmap
from jax.lax import scan
from functools import partial
from collections import namedtuple

from exact_simulator_utils import _get_local_rho, _apply_layer, _apply_control_signal, complete_system, M_inv

SimulatorState = namedtuple("SimulatorState", ["number_of_qubits", "system_qubit_number", "controlled_qubit_number", "iscrete_time"])


class ExactSimulator:

    def __init__(self) -> None:
        """[Class wrapping exact quantum dynamics simulator.]
        """
        pass

    def initialize(self,
                   number_of_qubits: int,
                   system_qubit_number: int,
                   controlled_qubit_number: int,
                   discrete_time: int) -> SimulatorState:
        """[This method initializes state of the simulator.]

        Args:
            number_of_qubits (int): [number of qubits in a system (must be odd)]
            system_qubit_number (int): [number of a qubit that is seen as the target system]
            controlled_qubit_number (int): [number of qubit under control]
            discrete_time (int): [number of time steps]

        Returns:
            SimulatorState: [state of the simulator]
        """

        state = SimulatorState(
            number_of_qubits,
            system_qubit_number,
            controlled_qubit_number,
            discrete_time
        )
        return state

    @partial(jit, static_argnums=(0, 1))
    def compute_dynamics_of_density_matrices(self,
                                             sim_state: SimulatorState,
                                             initial_system_state: jnp.ndarray,
                                             initial_environment_state: jnp.ndarray,
                                             gates_layer: jnp.ndarray,
                                             control_gates: jnp.ndarray) -> jnp.ndarray:
        """[This method compute dynamics of individual density matrices of
        all qubits under control.]

        Args:
            sim_state (SimulatorState): [state of the simulator]
            initial_system_state (complex valued jnp.ndarray of shape (2,))
                [state of the target qubit]
            initial_environment_state (complex valued jnp.ndarray of shape
                 (2 ** (number of qubits-1),)): [initial state of the environment]
            gates_layer (complex valued jnp.ndarray of shape (number of qubits-1, 4, 4)):
                [layer of gates describing dynamics of the whole system]
            control_gates (complex valued jnp.ndarray of shape (discrete_time, 2, 2)):
                [one-qubit control gates]
            controlled_qubit_number (int): [number of qubit under control]

        Returns:
            complex valued jnp.ndarray of shape (discrete_time, number_of_qubits, 2, 2): [dynamics of
            density matrices]
        """

        state = jnp.tensordot(initial_system_state, initial_environment_state, axes=0)
        state = state.reshape((2, 2 ** sim_state.system_qubit_number, -1))
        state = state.transpose((1, 0, 2))
        state = state.reshape((-1,))

        def iter(state, u):
            state = _apply_layer(gates_layer, state)
            state = _apply_control_signal(state, u, sim_state.controlled_qubit_number)
            rhos = _get_local_rho(state, sim_state.number_of_qubits)
            return state, rhos

        _, rhos = scan(iter, state, control_gates)
        return rhos

    @partial(jit, static_argnums=(0, 1))
    def compute_quantum_channels(self,
                                 sim_state: SimulatorState,
                                 initial_environment_state: jnp.ndarray,
                                 gates_layer: jnp.ndarray,
                                 control_gates: jnp.ndarray) -> jnp.ndarray:
        """[This method compute quantum channels that map initial state of
        a particular qubit to the state of all qubits at any discrete time moment.]

        Args:
            sim_state (SimulatorState): [state of the simulator]
            initial_environment_state (complex valued jnp.ndarray of shape
                 (2 ** (number of qubits-1),)): [initial state of the environment]
            gates_layer (complex valued jnp.ndarray of shape (number of qubits-1, 4, 4)):
                [layer of gates describing dynamics of the whole system]
            control_gates (complex valued jnp.ndarray of shape (discrete_time, 2, 2)):
                [one-qubit control gates]
            controlled_qubit_number (int): [number of qubit under control]

        Returns:
            jnp.ndarray: complex valued jnp.ndarray of shape (discrete_time, number_of_qubits, 2, 2, 2, 2):
            [quantum channels]
        """

        fun = vmap(self.compute_dynamics_of_density_matrices, in_axes=(None, 0, None, None, None), out_axes=-1)
        phi = fun(sim_state, complete_system, initial_environment_state, gates_layer, control_gates)
        phi = jnp.tensordot(phi, M_inv, axes=1)
        return phi
