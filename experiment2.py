import os
from functools import reduce
import uuid
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt

from simulators.logger import save_params, save_data
from simulators.models_utils import params2gates_layer
from simulators.exact_simulator import ExactSimulator
from simulators.reduced_order_simulator import ReducedOrderSimulator
from simulators.control import optimize, random_isometric
from simulators.exact_simulator_utils import sigma

# This file performs simulation of the system using both the exact
# simulator and the reduced-order simulator with zero control signal.

# ------------------ Description of model parameters ------------------------ #
# N --------------------- number of discrete time steps
# n --------------------- number of spins
# tau ------------------- time step duration
# hi -------------------- i-th component of the external magnetic field (i \in {x, y, z})
# Ji -------------------- i-th component of the coupling (i \in {x, y, z})
# system_qubit ---------- the number of the system spin (ranges from 0, to n-1)
# system_state ---------- psi vector of the system spin
# env_single_spin_state - psi vector of each spin in the environment
# eps ------------------- accuracy of truncation
# full_truncation ------- flag showing whether to use full truncation or not
# PRNGKey --------------- jax PRNGKey

set_of_params = {
    'N' : [50],
    'n' : [7],
    'tau' : [0.15],
    'hx' : [0.2],
    'hy' : [0.2],
    'hz' : [0.2],
    'Jx' : [0.9],
    'Jy' : [1],
    'Jz' : [1.1],
    'system_qubit' : [0],
    'system_state' : [[1, 0]],
    'env_single_spin_state' : [[0, 1]],
    'eps' : [1e-2],
    'full_truncation' : [True],
    'truncate_when' : [512],
    'random_seed' : [42],
}


def entropy(rho):
    spec = jnp.linalg.svd(rho, compute_uv=False)
    return -(spec * jnp.log(spec)).sum(-1).real
def mutual_information(phis):
    rho1 = jnp.trace(phis, axis1=-3, axis2=-4)
    rho2 = jnp.trace(phis, axis1=-1, axis2=-2)
    s1 = entropy(rho1)
    s2 = entropy(rho2)
    phis = jnp.swapaxes(phis, -3, -2)
    phis = phis.reshape((*phis.shape[:-4], 4, 4))
    s12 = entropy(phis)
    return s1 + s2 - s12


def run_experiment(set_of_params):
    set_of_params = [dict(zip(set_of_params.keys(), vals)) for vals in zip(*set_of_params.values())]
    for params in set_of_params:
        key = random.PRNGKey(params['random_seed'])
        experiment_id = str(uuid.uuid4())
        dir_path = 'experiment2_data/' + experiment_id
        os.mkdir(dir_path)

        save_params(params, dir_path + '/params.txt')

        # gates
        gates_layer = params2gates_layer(params)
        save_data(gates_layer, dir_path + '/gates_layer.pickle')

        # initial env. state for the reduced-order simulator
        ro_env_state = (params['n'] - 1) * [jnp.array(params['env_single_spin_state'], dtype=jnp.complex64)]

        # initial env. state for the exact simulator
        ex_env_state = reduce(lambda x, y: jnp.kron(x, y), ro_env_state)

        # system state
        system_state = jnp.array(params['system_state'], dtype=jnp.complex64)

        # zero control
        trivial_control_gates = jnp.tile(jnp.eye(2, dtype=jnp.complex64)[jnp.newaxis], (params['N'], 1, 1))
        control_gates = random_isometric(key, (params['N'], 2, 2))

        # EXACT DYNAMICS SIMULATION
        ex_sim = ExactSimulator()
        ex_sim_state = ex_sim.initialize(params['n'],
                                        params['system_qubit'],
                                        params['system_qubit'],
                                        params['N'])
        zero_control_quantum_channels = ex_sim.compute_quantum_channels(ex_sim_state,
                                                                        ex_env_state,
                                                                        gates_layer,
                                                                        trivial_control_gates)
        zero_control_exact_density_matrices = jnp.einsum('ijklmn,m,n->ijkl', zero_control_quantum_channels, system_state, system_state.conj())
        zero_control_mutual_information = mutual_information(zero_control_quantum_channels)
        save_data(zero_control_mutual_information, dir_path + '/zero_control_mutual_information.pickle')
        save_data(zero_control_quantum_channels, dir_path + '/zero_control_quantum_channels.pickle')
        save_data(zero_control_exact_density_matrices, dir_path + '/zero_control_exact_density_matrices.pickle')

        # REDUCED_ORDER MODEL BASED DYNAMICS SIMULATION
        ro_sim = ReducedOrderSimulator()
        ro_model = ro_sim.build_reduced_order_model(params['system_qubit'],
                                                    params['system_qubit'],
                                                    params['N'],
                                                    ro_env_state,
                                                    gates_layer,
                                                    params['full_truncation'],
                                                    params['truncate_when'],
                                                    params['eps'])
        save_data(ro_model, dir_path + '/ro_model.pickle')
        zero_control_ro_model_based_density_matrices = ro_sim.compute_dynamics(ro_model, trivial_control_gates, system_state)
        save_data(zero_control_ro_model_based_density_matrices, dir_path + '/zero_control_ro_based_density_matrices.pickle')

        # CONTROL SIGNAL OPTIMIZATION
        def loss_fn(ro_model, control_gates):
            half_N = params['N'] // 2
            phis = ro_sim.compute_quantum_channels(ro_model, control_gates)
            id_channel = jnp.eye(4).reshape((2, 2, 2, 2))
            complete_depolarizing = 0.5 * jnp.tensordot(jnp.eye(2), jnp.eye(2), axes=0)
            return jnp.linalg.norm(id_channel - phis[-1]) ** 2 + jnp.linalg.norm(complete_depolarizing - phis[half_N]) ** 2
        control_gates, learning_curve = optimize(loss_fn, ro_model, control_gates, 10, 2000, 0.02)
        save_data(control_gates, dir_path + '/control_gates.pickle')
        save_data(learning_curve, dir_path + '/learning_curve.pickle')

        # EXACT DYNAMICS SIMULATION WITH CONTROL
        controlled_quantum_channels = ex_sim.compute_quantum_channels(ex_sim_state,
                                                                     ex_env_state,
                                                                     gates_layer,
                                                                     control_gates)
        controlled_mutual_information = mutual_information(controlled_quantum_channels)
        controlled_exact_density_matrices = jnp.einsum('ijklmn,m,n->ijkl', controlled_quantum_channels, system_state, system_state.conj())
        save_data(controlled_quantum_channels, dir_path + '/controlled_quantum_channels.pickle')
        save_data(controlled_exact_density_matrices, dir_path + '/controlled_exact_density_matrices.pickle')
        save_data(controlled_mutual_information, dir_path + '/controlled_mutual_information.pickle')

        # REDUCED_ORDER MODEL BASED DYNAMICS SIMULATION WITH CONTROL
        controlled_ro_model_based_density_matrices = ro_sim.compute_dynamics(ro_model, control_gates, system_state)
        save_data(controlled_ro_model_based_density_matrices, dir_path + '/controlled_ro_based_density_matrices.pickle')


        #SIMPLE PLOTTING

        zero_control_ro_bloch_vectors = jnp.tensordot(zero_control_ro_model_based_density_matrices, sigma, axes=[[1, 2], [2, 1]])
        zero_control_exact_bloch_vectors = jnp.tensordot(zero_control_exact_density_matrices[:, params['system_qubit']], sigma, axes=[[1, 2], [2, 1]])
        controlled_ro_bloch_vectors = jnp.tensordot(controlled_ro_model_based_density_matrices, sigma, axes=[[1, 2], [2, 1]])
        controlled_exact_bloch_vectors = jnp.tensordot(controlled_exact_density_matrices[:, params['system_qubit']], sigma, axes=[[1, 2], [2, 1]])

        plt.figure()
        plt.plot(zero_control_ro_bloch_vectors[:, 0], 'r')
        plt.plot(zero_control_ro_bloch_vectors[:, 1], 'b')
        plt.plot(zero_control_ro_bloch_vectors[:, 2], 'k')
        plt.plot(zero_control_exact_bloch_vectors[:, 0], '*r')
        plt.plot(zero_control_exact_bloch_vectors[:, 1], 'ob')
        plt.plot(zero_control_exact_bloch_vectors[:, 2], 'xk')
        plt.legend(['exact x', 'exact y', 'exact z', 'ro x', 'ro y', 'ro z'])
        plt.ylabel('Amplitude')
        plt.xlabel('N')
        plt.savefig(dir_path + '/zero_control_dynamics.pdf')

        plt.figure()
        plt.plot(controlled_ro_bloch_vectors[:, 0], 'r')
        plt.plot(controlled_ro_bloch_vectors[:, 1], 'b')
        plt.plot(controlled_ro_bloch_vectors[:, 2], 'k')
        plt.plot(controlled_exact_bloch_vectors[:, 0], '*r')
        plt.plot(controlled_exact_bloch_vectors[:, 1], 'ob')
        plt.plot(controlled_exact_bloch_vectors[:, 2], 'xk')
        plt.legend(['exact x', 'exact y', 'exact z', 'ro x', 'ro y', 'ro z'])
        plt.ylabel('Amplitude')
        plt.xlabel('N')
        plt.savefig(dir_path + '/controlled_dynamics.pdf')

        plt.figure()
        plt.plot([ker.shape[0] * ker.shape[2] for ker in reversed(ro_model)])
        plt.ylabel('Environment dimension')
        plt.xlabel('N')
        plt.savefig(dir_path + '/env_dimension.pdf')

        plt.figure()
        plt.plot(learning_curve, 'b')
        plt.ylabel('loss_value')
        plt.xlabel('epoch_number')
        plt.savefig(dir_path + '/learning_curve.pdf')

        plt.figure()
        plt.imshow(zero_control_mutual_information, cmap='inferno')
        plt.xlabel('spin_number')
        plt.ylabel('N')
        plt.savefig(dir_path + '/zero_control_mutual_information.pdf')

        plt.figure()
        plt.imshow(controlled_mutual_information, cmap='inferno')
        plt.xlabel('spin_number')
        plt.ylabel('N')
        plt.savefig(dir_path + '/controlled_mutual_information.pdf')

run_experiment(set_of_params)
