from jax.config import config
from jax.core import lattice_join
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

import pytest

from jax import numpy as jnp
from jax import random

from simulators.reduced_order_simulator import ReducedOrderSimulator
from simulators.exact_simulator import ExactSimulator

from functools import reduce

key = random.PRNGKey(42)
key, subkey = random.split(key)

@pytest.mark.parametrize("system_qubit,number_of_qubits", [(3, 7), (0, 7), (6, 7)])
def test_reduced_order_simulator(system_qubit, number_of_qubits):

    controlled_qubit_number = system_qubit
    eps = 1e-10
    truncate_when = 3
    discrete_time = 10


    # random gates layer
    gates_layer = random.normal(key, (number_of_qubits - 1, 4, 4, 2))
    gates_layer = gates_layer[..., 0] + 1j * gates_layer[..., 1]
    gates_layer, _ = jnp.linalg.qr(gates_layer)

    # random control gates
    control_gates = random.normal(key, (discrete_time, 2, 2, 2))
    control_gates = control_gates[..., 0] + 1j * control_gates[..., 1]
    control_gates, _ = jnp.linalg.qr(control_gates)

    # initial state of the environment (input for the reduced order simulator)
    in_env_state = (number_of_qubits - 1) * [jnp.array([1, 0], dtype=jnp.complex128)]

    # initial state of the system
    in_system_state = jnp.array([0, 1], dtype=jnp.complex128)

    # initial state of the environment (input for the exact simulator)
    in_env_state_for_exact_sim = reduce(lambda x, y: jnp.kron(x, y), in_env_state)

    ro_sim = ReducedOrderSimulator()
    ex_sim = ExactSimulator()

    sim_state = ex_sim.initialize(number_of_qubits,
                                system_qubit,
                                controlled_qubit_number,
                                discrete_time)

    ex_rhos = ex_sim.compute_dynamics_of_density_matrices(sim_state,
                                                          in_system_state,
                                                          in_env_state_for_exact_sim,
                                                          gates_layer,
                                                          control_gates)

    ro_model = ro_sim.build_reduced_order_model(system_qubit,
                                                controlled_qubit_number,
                                                discrete_time,
                                                in_env_state,
                                                gates_layer,
                                                truncate_when,
                                                eps)

    ro_rhos = ro_sim.compute_dynamics(ro_model, control_gates, in_system_state)

    assert jnp.linalg.norm(ex_rhos[:, system_qubit] - ro_rhos) < 1e-10, "Results of exact and reduced-order simulations do not match"
