import os #os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\bin")
import jax
jax.config.update('jax_platform_name', 'cpu')
#from jax.config import config
#config.update("jax_enable_x64", True)

from dataclasses import asdict
import uuid
import jax.numpy as jnp
from jax import random
from datetime import datetime
import matplotlib.pyplot as plt

from simulators.logger import save_params, save_data
from simulators.models_utils import params2gates_layer, energy_dist
from simulators.exact_simulator import ExactSimulator
from simulators.reduced_order_simulator import ReducedOrderSimulator
from simulators.control import optimize
from simulators.dataclasses import ExperimentParameters
from experiments_utils import rom2exact_init_state_converter, zero_control_seq, random_control_seq, channels2rhos, rho2bloch, mutual_information
from environment_energy import params2hamiltonian_mpo, renorm_hamiltonian, environment_energy


set_of_params = ExperimentParameters(
        N = (100,),
        n = (21,),
        tau = (0.05,),
        hx = (0.3,),
        hy = (0.2,),
        hz = (0.1,),
        Jx = (0.9,),
        Jy = (1.,),
        Jz = (1.1,),
        system_qubit = (0,),
        source_qubit = (None,),  # this parameter is not active in this experiment
        system_state = ([1, 0],),
        env_single_spin_state = ([0, 1],),
        eps = (1e-3,),
        startN = (None,),  # this parameter is not active in this experiment
        stopN = (None,),  # this parameter is not active in this experiment
        full_truncation = (False,),
        truncate_when = (64,),
        random_seed = (314,),
        learning_rate = (0.03,),
        epoch_size = (100,),
        number_of_epoches = (100,),
        fast_jit =  (False,),
)


def run_experiment(set_of_params: ExperimentParameters):
    i = 0
    for params in zip(*set_of_params):
        params = ExperimentParameters(*params)

        print("Environment cooling experiment started")
        key = random.PRNGKey(params.random_seed)

        print("Subexperiment #{} is run. \n".format(i+1))
        experiment_id = str(datetime.now())
        dir_path = 'experiment_cooling_data/' + experiment_id
        os.mkdir(dir_path)

        # Save parameters
        save_params(asdict(params), dir_path + '/params.txt')
        # Gates
        gates_layer, ham_layer = params2gates_layer(params)
        save_data(gates_layer, dir_path + '/gates_layer.pickle')
        save_data(ham_layer, dir_path + '/ham_layer.pickle')
        print("Gates layer generated")

        # Initial env. state for ROM simulator
        ro_env_state = (params.n - 1) * [jnp.array(params.env_single_spin_state,
                                                   dtype=jnp.complex64)]
        print("ROM environment state: INIT")

        # Initial env. state for exact simulator
        ex_env_state = rom2exact_init_state_converter(ro_env_state)
        print("Exact environment state: INIT")

        # System state
        system_state = jnp.array(params.system_state, dtype=jnp.complex64)
        print("Exact system state: INIT")

        # Zero control seq.
        trivial_control_gates = zero_control_seq(params.N)
        print("Trivial control gates: INIT")

        # Initial non-zero control seq.
        control_gates = random_control_seq(key, params.N)
        print("Random control gates: INIT")

        ##################################################
        # EXACT DYNAMICS SIMULATION
        ex_sim = ExactSimulator()
        ex_sim_state = ex_sim.initialize(
            params.n,
            params.system_qubit,
            params.system_qubit,
            params.N,
        )

        zero_control_exact_loc_density_matrices, \
        zero_control_exact_two_density_matrices = ex_sim.compute_dynamics_of_density_matrices(
                                                                        ex_sim_state,
                                                                        system_state,
                                                                        ex_env_state,
                                                                        gates_layer,
                                                                        trivial_control_gates)

        # LOGGING SIMULATED DATA
        save_data(zero_control_exact_loc_density_matrices, dir_path + \
                    '/zero_control_exact_local_density_matrices.pickle')
        save_data(zero_control_exact_two_density_matrices, dir_path + \
                    '/zero_control_exact_twolocal_density_matrices.pickle')
        print("Exact simulation under zero control: DONE")
        print("Subexperiment #{}".format(i+1))

        ##################################################
        # ZERO CONTROL ENERGY FLOW
        zero_control_energy_distribution = energy_dist(ham_layer,
                                           zero_control_exact_two_density_matrices)
        save_data(zero_control_energy_distribution, dir_path + \
                    '/zero_control_energy_distribution.pickle')
        print("Zero control energy flow: DONE")
        ##################################################
        # REDUCED_ORDER MODEL BASED DYNAMICS SIMULATION
        ro_sim = ReducedOrderSimulator()
        ro_model, isometries, _ = ro_sim.build_reduced_order_model(
                                            params.system_qubit,
                                            params.system_qubit,
                                            params.N,
                                            ro_env_state,
                                            gates_layer,
                                            params.full_truncation,
                                            params.truncate_when,
                                            params.eps)

        zero_control_ro_density_matrices, ro_states = ro_sim.compute_dynamics(
                                                        ro_model,
                                                        trivial_control_gates,
                                                        system_state)

        # LOGGING SIMULATED DATA
        #save_data(ro_model, dir_path + '/ro_model.pickle')
        save_data(zero_control_ro_density_matrices, dir_path +\
                  '/zero_control_reduced_order_density_matrices.pickle')
        print("Reduced-order model: DONE")
        print("Subexperiment #{}".format(i+1))
        print("RUNNING Control sequence optimization:")

        ##################################################
        # CONTROL SIGNAL OPTIMIZATION
        if params.fast_jit:
            ro_model = ro_sim.preprocess_reduced_order_model(ro_model)

        # Transform parameters
        couplings = jnp.array((params.n - 1) *
                              [params.Jx,
                               params.Jy,
                               params.Jz]).reshape(
                               params.n - 1, 3)

        fields = jnp.array((params.n) *
                              [params.hx,
                               params.hy,
                               params.hz]).reshape(
                               params.n, 3)

        top_isometries, bot_isometries = isometries
        resh_bot_iso = list(map(list, zip(*bot_isometries)))
        resh_top_iso = list(map(list, zip(*top_isometries)))

        # Hamiltonian renormalization
        mpo_hamiltonian = params2hamiltonian_mpo(couplings, fields)
        print('Hamiltonian MPO: DONE')

        if params.system_qubit == 0:
            last_iso = ([], resh_bot_iso[-1])
        elif params.system_qubit == params.n - 1:
            last_iso = (resh_top_iso[-1], [])
        else:
            last_iso = (resh_top_iso[-1], resh_bot_iso[-1])

        renormalized_ham = renorm_hamiltonian(mpo_hamiltonian,
                                              last_iso,
                                              params.system_qubit)
        print('Hamiltonian renormalization: DONE')

        # Loss function that is being optimized
        def loss_function(ro_model, control_gates):
            _, ro_states = ro_sim.compute_dynamics(ro_model, control_gates, system_state)
            return environment_energy(renormalized_ham, ro_states[-1])

        print("Optimization process START:")
        control_gates, learning_curve = optimize(
                                loss_function,
                                ro_model,
                                control_gates,
                                params.number_of_epoches,
                                params.epoch_size,
                                params.learning_rate,
        )

        # LOGGING SIMULATED DATA
        save_data(control_gates, dir_path + '/control_gates.pickle')
        save_data(learning_curve, dir_path + '/learning_curve.pickle')

        print("Control sequence optimization: DONE")
        print("Subexperiment #{}".format(i+1))


        ##################################################
        # EXACT DYNAMICS SIMULATION WITH CONTROL
        controlled_quantum_channels = ex_sim.compute_quantum_channels(
            ex_sim_state,
            ex_env_state,
            gates_layer,
            control_gates
        )
        controlled_mutual_information = mutual_information(controlled_quantum_channels)
        controlled_exact_density_matrices = channels2rhos(controlled_quantum_channels, system_state)

        # LOGGING SIMULATED DATA
        save_data(controlled_quantum_channels, dir_path + '/controlled_quantum_channels.pickle')
        save_data(controlled_exact_density_matrices, dir_path + '/controlled_exact_density_matrices.pickle')
        save_data(controlled_mutual_information, dir_path + '/controlled_mutual_information.pickle')

        # REDUCED_ORDER MODEL BASED DYNAMICS SIMULATION WITH CONTROL
        controlled_ro_density_matrices, controlled_states = ro_sim.compute_dynamics(ro_model,
                                                                                control_gates,
                                                                                system_state)

        # LOGGING SIMULATED DATA
        save_data(controlled_ro_density_matrices, dir_path + '/controlled_ro_based_density_matrices.pickle')
        print("Exact dynamics simulation under optimal control sequence for subexperiment #{}: DONE.".format(i+1))

        ##################################################
        # CONTROLLED ENEGY FLOW

        controlled_exact_loc_density_matrices, \
        controlled_exact_two_density_matrices = ex_sim.compute_dynamics_of_density_matrices(
                                                                        ex_sim_state,
                                                                        system_state,
                                                                        ex_env_state,
                                                                        gates_layer,
                                                                        control_gates)
        controlled_energy_distribution = energy_dist(ham_layer,
                                           controlled_exact_two_density_matrices)
        save_data(controlled_energy_distribution, dir_path + \
                    '/controlled_energy_distribution.pickle')
        print("Controlled energy flow: DONE")

        ##################################################
        #SIMPLE PLOTTING

        zero_control_ro_bloch_vectors = rho2bloch(zero_control_ro_density_matrices)
        zero_control_exact_bloch_vectors = rho2bloch(zero_control_exact_loc_density_matrices[:, params.system_qubit])
        controlled_ro_bloch_vectors = rho2bloch(controlled_ro_density_matrices)
        controlled_exact_bloch_vectors = rho2bloch(controlled_exact_density_matrices[:, params.system_qubit])

        plt.figure(figsize=(15, 5))
        plt.imshow(zero_control_energy_distribution, cmap='inferno_r')
        plt.savefig(dir_path + '/zero_control_energy_distribution.pdf')

        plt.figure(figsize=(15, 5))
        plt.imshow(controlled_energy_distribution, cmap='inferno_r')
        plt.savefig(dir_path + '/controlled_energy_distribution.pdf')

        plt.figure(figsize=(12, 4))
        plt.plot(zero_control_ro_bloch_vectors[:, 0], 'or', markersize=4)
        plt.plot(zero_control_ro_bloch_vectors[:, 1], 'ob', markersize=4)
        plt.plot(zero_control_ro_bloch_vectors[:, 2], 'og', markersize=4)
        plt.plot(zero_control_exact_bloch_vectors[:, 0], '--r')
        plt.plot(zero_control_exact_bloch_vectors[:, 1], '--b')
        plt.plot(zero_control_exact_bloch_vectors[:, 2], '--g')
        plt.legend(['Exact x', 'Exact y', 'Exact z', 'RO x', 'RO y', 'RO z'])
        plt.ylabel('Amplitude')
        plt.xlabel('N')
        plt.savefig(dir_path + '/zero_control_dynamics.pdf')

        plt.figure(figsize=(12, 4))
        plt.plot(controlled_ro_bloch_vectors[:, 0], 'or', markersize=4)
        plt.plot(controlled_ro_bloch_vectors[:, 1], 'ob', markersize=4)
        plt.plot(controlled_ro_bloch_vectors[:, 2], 'ok', markersize=4)
        plt.plot(controlled_exact_bloch_vectors[:, 0], '--r')
        plt.plot(controlled_exact_bloch_vectors[:, 1], '--b')
        plt.plot(controlled_exact_bloch_vectors[:, 2], '--g')
        plt.legend(['Exact x', 'Exact y', 'Exact z', 'RO x', 'RO y', 'RO z'])
        plt.ylabel('Amplitude')
        plt.xlabel('N')
        plt.savefig(dir_path + '/controlled_dynamics.pdf')

        plt.figure()
        plt.plot([2 * rom_kernel.ker_top.shape[0] * rom_kernel.ker_bottom.shape[0] for rom_kernel in reversed(ro_model)])
        plt.ylabel('Environment dimension')
        plt.xlabel('N')
        plt.yscale('log')
        plt.savefig(dir_path + '/env_dimension.pdf')

        plt.figure()
        plt.plot(learning_curve, 'b')
        plt.ylabel('loss_value')
        plt.xlabel('epoch_number')
        plt.savefig(dir_path + '/learning_curve.pdf')
        i += 1

run_experiment(set_of_params)
