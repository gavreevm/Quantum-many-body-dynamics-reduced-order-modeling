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
from simulators.exact_simulator_utils import complete_system


# This file performs simulation of the information transition from the
# source qubit to the system qubit

# ------------------ Description of model parameters ------------------------ #
# N --------------------- number of discrete time steps
# n --------------------- number of spins
# tau ------------------- time step duration
# hi -------------------- i-th component of the external magnetic field (i \in {x, y, z})
# Ji -------------------- i-th component of the coupling (i \in {x, y, z})
# system_qubit ---------- the number of the system spin (ranges from 0, to n-1)
# source_qubit ---------- the number of the source spin (ranges from 0, to n-1)
# env_single_spin_state - psi vector of each spin in the environment
# eps ------------------- accuracy of truncation
# full_truncation ------- flag showing whether to use full truncation or not
# truncate_when --------- the bond dim. value threshold
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
    'system_qubit' : [1],
    'source_qubit' : [5],
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

        # initial env. states for the reduced-order simulator
        ro_env_state0 = (params['n'] - 2) * [jnp.array(params['env_single_spin_state'], dtype=jnp.complex64)]
        ro_env_state0.insert(params['source_qubit']-1, complete_system[0])
        ro_env_state1 = (params['n'] - 2) * [jnp.array(params['env_single_spin_state'], dtype=jnp.complex64)]
        ro_env_state1.insert(params['source_qubit']-1, complete_system[1])
        ro_env_state2 = (params['n'] - 2) * [jnp.array(params['env_single_spin_state'], dtype=jnp.complex64)]
        ro_env_state2.insert(params['source_qubit']-1, complete_system[2])
        ro_env_state3 = (params['n'] - 2) * [jnp.array(params['env_single_spin_state'], dtype=jnp.complex64)]
        ro_env_state3.insert(params['source_qubit']-1, complete_system[3])

        # system state
        system_state = jnp.array(params['env_single_spin_state'], dtype=jnp.complex64)

        # initial env. state for the exact simulator
        ex_env_state = reduce(lambda x, y: jnp.kron(x, y), (params['n'] - 1) * [system_state])

        # initial control seq.
        control_gates = random_isometric(key, (params['N'], 2, 2))

        # REDUCED_ORDER MODEL BASED DYNAMICS SIMULATION
        ro_sim = ReducedOrderSimulator()
        ro_model0 = ro_sim.build_reduced_order_model(params['system_qubit'],
                                                     params['system_qubit'],
                                                     params['N'],
                                                     ro_env_state0,
                                                     gates_layer,
                                                     params['full_truncation'],
                                                     params['truncate_when'],
                                                     params['eps'])
        ro_model1 = ro_sim.build_reduced_order_model(params['system_qubit'],
                                                     params['system_qubit'],
                                                     params['N'],
                                                     ro_env_state1,
                                                     gates_layer,
                                                     params['full_truncation'],
                                                     params['truncate_when'],
                                                     params['eps'])
        ro_model2 = ro_sim.build_reduced_order_model(params['system_qubit'],
                                                     params['system_qubit'],
                                                     params['N'],
                                                     ro_env_state2,
                                                     gates_layer,
                                                     params['full_truncation'],
                                                     params['truncate_when'],
                                                     params['eps'])
        ro_model3 = ro_sim.build_reduced_order_model(params['system_qubit'],
                                                     params['system_qubit'],
                                                     params['N'],
                                                     ro_env_state3,
                                                     gates_layer,
                                                     params['full_truncation'],
                                                     params['truncate_when'],
                                                     params['eps'])
        ro_models = [
            ro_model0,
            ro_model1,
            ro_model2,
            ro_model3
        ]
        save_data(ro_models, dir_path + '/ro_models.pickle')

        # CONTROL SIGNAL OPTIMIZATION
        def loss_fn(ro_models, control_gates):
            ro_model0, ro_model1, ro_model2, ro_model3 = ro_models
            rho0 = ro_sim.compute_dynamics(ro_model0, control_gates, system_state)[-1]
            rho1 = ro_sim.compute_dynamics(ro_model1, control_gates, system_state)[-1]
            rho2 = ro_sim.compute_dynamics(ro_model2, control_gates, system_state)[-1]
            rho3 = ro_sim.compute_dynamics(ro_model3, control_gates, system_state)[-1]
            rho_inp0 = jnp.tensordot(complete_system[0], complete_system[0].conj(), axes=0)
            rho_inp1 = jnp.tensordot(complete_system[1], complete_system[1].conj(), axes=0)
            rho_inp2 = jnp.tensordot(complete_system[2], complete_system[2].conj(), axes=0)
            rho_inp3 = jnp.tensordot(complete_system[3], complete_system[3].conj(), axes=0)
            loss_val = (
                jnp.linalg.norm(rho0 - rho_inp0) ** 2 +
                jnp.linalg.norm(rho1 - rho_inp1) ** 2 +
                jnp.linalg.norm(rho2 - rho_inp2) ** 2 +
                jnp.linalg.norm(rho3 - rho_inp3) ** 2
            )
            return loss_val


        control_gates, learning_curve = optimize(loss_fn, ro_models, control_gates, 10, 2000, 0.01)
        save_data(control_gates, dir_path + '/control_gates.pickle')
        save_data(learning_curve, dir_path + '/learning_curve.pickle')

        # STATE TRANSFER EXACT SIMULATION
        ex_sim = ExactSimulator()
        ex_sim_state = ex_sim.initialize(params['n'],
                                         params['source_qubit'],
                                         params['system_qubit'],
                                         params['N'])
        controlled_quantum_channels = ex_sim.compute_quantum_channels(ex_sim_state,
                                                                      ex_env_state,
                                                                      gates_layer,
                                                                      control_gates)
        controlled_mutual_information = mutual_information(controlled_quantum_channels)
        controlled_exact_density_matrices = jnp.einsum('ijklmn,m,n->ijkl', controlled_quantum_channels, system_state, system_state.conj())
        save_data(controlled_quantum_channels, dir_path + '/controlled_quantum_channels.pickle')
        save_data(controlled_exact_density_matrices, dir_path + '/controlled_exact_density_matrices.pickle')
        save_data(controlled_mutual_information, dir_path + '/controlled_mutual_information.pickle')

        #SIMPLE PLOTTING

        plt.figure()
        plt.plot(learning_curve, 'b')
        plt.ylabel('loss_value')
        plt.xlabel('epoch_number')
        plt.savefig(dir_path + '/learning_curve.pdf')

        plt.figure()
        plt.imshow(controlled_mutual_information, cmap='inferno')
        plt.xlabel('spin_number')
        plt.ylabel('N')
        plt.savefig(dir_path + '/controlled_mutual_information.pdf')

run_experiment(set_of_params)
