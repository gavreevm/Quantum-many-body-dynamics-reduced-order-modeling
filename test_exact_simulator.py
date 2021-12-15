from jax.config import config
config.update("jax_enable_x64", True)

from jax import numpy as jnp
from jax.scipy.linalg import expm
from exact_simulator import ExactSimulator
from jax import random

import matplotlib.pyplot as plt


def get_random_initial_states_of_sys_and_env(key, number_of_qubits):
    key, subkey = random.split(key)
    initial_environment_state = random.normal(key, (2 ** (number_of_qubits-1), 2))
    initial_environment_state = initial_environment_state[:, 0] + 1j * initial_environment_state[:, 1]
    initial_environment_state /= jnp.linalg.norm(initial_environment_state)

    initial_system_state = random.normal(subkey, (2, 2))
    initial_system_state = initial_system_state[:, 0] + 1j * initial_system_state[:, 1]
    initial_system_state /= jnp.linalg.norm(initial_system_state)
    return initial_system_state, initial_environment_state


def test_ExactSimulator_compute_dynamics_of_density_matrices():
    """This is a spin echo test, i.e. we implement spin echo protocol and check
    if it works with the given simulator"""
    number_of_qubits = 7
    free_time = 5
    system_qubit_number = 3
    controled_qubit_number = system_qubit_number

    discrete_time = 2 * free_time + 2

    key = random.PRNGKey(42)

    sigma_z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)
    sigma_x = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
    hzz = jnp.tensordot(sigma_z, sigma_z, axes=0)
    hzz = hzz.transpose((0, 2, 1, 3))
    hzz = hzz.reshape((4, 4))
    gates_layer = expm(-1j * 0.3 * hzz)
    gates_layer = jnp.tile(gates_layer[jnp.newaxis], (number_of_qubits-1, 1, 1))

    initial_system_state, initial_environment_state = get_random_initial_states_of_sys_and_env(key, number_of_qubits)

    control_gates = jnp.tile(jnp.eye(2)[jnp.newaxis], (free_time, 1, 1))
    control_gates = jnp.concatenate([control_gates, sigma_x[jnp.newaxis], control_gates], axis=0)

    sim = ExactSimulator()
    sim_state = sim.initialize(number_of_qubits,
                               system_qubit_number,
                               controled_qubit_number,
                               discrete_time)

    rhos = sim.compute_dynamics_of_density_matrices(sim_state,
                                                    initial_system_state,
                                                    initial_environment_state,
                                                    gates_layer,
                                                    control_gates)
    before_control = rhos[:free_time, system_qubit_number]
    after_control = rhos[-1:free_time:-1, system_qubit_number]
    err = jnp.linalg.norm(before_control - sigma_x @ after_control @ sigma_x)
    assert err, "Dynamics is not inverted in time for spin-echo protocol, the simulatro does not work correctly"


def test_ExactSimulator_compute_quantum_channels():
    """Here we simulate a random circuit, compute density matrices dynamics and quantum channels and
    finaly use quantum chennals to get density matrix dynamics. If the two ways of density matrix dynamics
    computation lead to different results then the method compute_quantum_channels is incorrect."""

    key = random.PRNGKey(43)
    key, subkey1, subkey2 = random.split(key, 3)

    number_of_qubits = 7
    discrete_time = 5
    system_qubit_number = 3
    controled_qubit_number = system_qubit_number

    initial_system_state, initial_environment_state = get_random_initial_states_of_sys_and_env(key, number_of_qubits)

    random_gates_set = random.normal(subkey1, (number_of_qubits-1, 4, 4, 2))
    random_gates_set = random_gates_set[..., 0] + 1j * random_gates_set[..., 1]
    random_gates_set, _ = jnp.linalg.qr(random_gates_set)

    control_gates = random.normal(subkey1, (discrete_time, 2, 2, 2))
    control_gates = control_gates[..., 0] + 1j * control_gates[..., 1]
    control_gates, _ = jnp.linalg.qr(control_gates)

    sim = ExactSimulator()
    sim_state = sim.initialize(number_of_qubits,
                               system_qubit_number,
                               controled_qubit_number,
                               discrete_time)

    rhos = sim.compute_dynamics_of_density_matrices(sim_state,
                                                    initial_system_state,
                                                    initial_environment_state,
                                                    random_gates_set,
                                                    control_gates)

    phis = sim.compute_quantum_channels(sim_state,
                                        initial_environment_state,
                                        random_gates_set,
                                        control_gates)

    rho_inp = jnp.tensordot(initial_system_state, initial_system_state.conj(), axes=0)
    rhos_via_phis = jnp.einsum('ijklmn,mn->ijkl', phis, rho_inp)
    err = jnp.linalg.norm(rhos_via_phis - rhos)
    assert err < 1e-10, 'Channels based computation of the density matrices dynamics is incorrect'
