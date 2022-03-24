import os
#os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\bin")
import jax
jax.config.update('jax_platform_name', 'cpu')

from dataclasses import asdict
import uuid
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt

from simulators.logger import save_params, save_data
from simulators.models_utils import sample_disordered_floquet
from simulators.exact_simulator import ExactSimulator
from simulators.reduced_order_simulator import ReducedOrderSimulator
from simulators.control import optimize
from simulators.dataclasses import ExperimentParameters
from experiments_utils import (
    mutual_information,
    rom2exact_init_state_converter,
    zero_control_seq,
    random_control_seq,
    channels2rhos,
    controll_padding,
)


# This file performs simulation of the MBL subsystem using both the exact
# simulator and the reduced-order simulator without control signal
# and with the optimal control.

# ------------------ Description of model parameters ------------------------ #
# N --------------------- number of discrete time steps
# n --------------------- number of spins
# tau ------------------- time step duration
# hx -------------------- strength of the transverse field component
# Jz -------------------- zz coupling strength
# system_qubit ---------- the number of the system spin (ranges from 0, to n-1)
# system_state ---------- psi vector of the system spin
# env_single_spin_state - psi vector of each spin in the environment
# eps ------------------- accuracy of truncation
# startN ---------------- the discrete time moment showing when to run the control sequance
# stopN ----------------- the discrete time moment showing when to stop the control sequance
# full_truncation ------- flag showing whether to use full truncation or not
# truncate_when --------- the bond dim. value threshold
# random_seed ----------- random seed
# learning_rate --------- learning rate that is used while searching of the optimal control sequance
# epoch_size ------------ optimization epoch size (number of iteration per epoch)
# number_of_epoches ----- total number of optimization epoches
# fast_jit -------------- ON/OFF fast_jit


# experiment parameters
set_of_params = ExperimentParameters(
    N = (151,),
    n = (21,),
    tau = (None,),  # this parameter is not active in this experiment
    hx = (0.2,),
    hy = (None,),  # this parameter is not active in this experiment
    hz = (None,),  # this parameter is not active in this experiment
    Jx = (None,),  # this parameter is not active in this experiment
    Jy = (None,),  # this parameter is not active in this experiment
    Jz = (0.2,),
    system_qubit = (0,),
    source_qubit = (None,),  # this parameter is not active in this experiment
    system_state = ([1, 0],),
    env_single_spin_state = ([0, 1],),
    eps = (5e-3,),
    startN = (50,),
    stopN = (101,),
    full_truncation = (True,),
    truncate_when = (512,),
    random_seed = (51,),
    learning_rate = (0.01,),
    epoch_size = (100,),
    number_of_epoches = (100,),
    fast_jit = (True,),
)


def run_experiment(set_of_params: ExperimentParameters):
    i = 0
    for params in zip(*set_of_params):
        params = ExperimentParameters(*params)
        
        control_duration = params.stopN - params.startN

        print("Subexperiment #{} is run. \n".format(i+1))

        key = random.PRNGKey(params.random_seed)
        experiment_id = str(uuid.uuid4())
        dir_path = 'experiment3_data_av/' + experiment_id
        os.mkdir(dir_path)

        save_params(asdict(params), dir_path + '/params.txt')

        # gates
        gates_layer = sample_disordered_floquet(params)
        save_data(gates_layer, dir_path + '/gates_layer.pickle')

        # initial env. state for the reduced-order simulator
        ro_env_state = (params.n - 1) * [jnp.array(params.env_single_spin_state, dtype=jnp.complex64)]

        # initial env. state for the exact simulator
        ex_env_state = rom2exact_init_state_converter(ro_env_state)

        # system state
        system_state = jnp.array(params.system_state, dtype=jnp.complex64)

        # zero control seq.
        trivial_control_gates = zero_control_seq(params.N)

        # initial non-zero control seq.
        control_gates = random_control_seq(key, control_duration)

        # EXACT DYNAMICS SIMULATION
        ex_sim = ExactSimulator()
        ex_sim_state = ex_sim.initialize(
            params.n,
            params.system_qubit,
            params.system_qubit,
            params.N,
        )
        zero_control_quantum_channels = ex_sim.compute_quantum_channels(
            ex_sim_state,
            ex_env_state,
            gates_layer,
            trivial_control_gates,
        )
        zero_control_exact_density_matrices = channels2rhos(zero_control_quantum_channels, system_state)
        zero_control_mutual_information = mutual_information(zero_control_quantum_channels)

        # LOGGING SIMULATED DATA
        save_data(zero_control_mutual_information, dir_path + '/zero_control_mutual_information.pickle')
        save_data(zero_control_quantum_channels, dir_path + '/zero_control_quantum_channels.pickle')
        save_data(zero_control_exact_density_matrices, dir_path + '/zero_control_exact_density_matrices.pickle')
        
        print("Exact dynamics simulation under zero control sequence for subexperiment #{} is done.".format(i+1))

        # REDUCED_ORDER MODEL BASED DYNAMICS SIMULATION
        ro_sim = ReducedOrderSimulator()
        ro_model = ro_sim.build_reduced_order_model(
            params.system_qubit,
            params.system_qubit,
            params.N,
            ro_env_state,
            gates_layer,
            params.full_truncation,
            params.truncate_when,
            params.eps)
        zero_control_ro_model_based_density_matrices = ro_sim.compute_dynamics(ro_model, trivial_control_gates, system_state)
        
        # LOGGING SIMULATED DATA
        save_data(ro_model, dir_path + '/ro_model.pickle')
        save_data(zero_control_ro_model_based_density_matrices, dir_path + '/zero_control_ro_based_density_matrices.pickle')
        
        print("Reduced-order model for subexperiment #{} is built.".format(i+1))
        print("Control sequance optimization is run:")

        ro_dim_vs_time = [2 * rom_kernel.ker_top.shape[0] * rom_kernel.ker_bottom.shape[0] for rom_kernel in reversed(ro_model)]
        save_data(ro_dim_vs_time, dir_path + '/ro_dimension_vs_time.pickle')

        # CONTROL SIGNAL OPTIMIZATION
        if params.fast_jit:
            ro_model = ro_sim.preprocess_reduced_order_model(ro_model)
            
        # This is the loss function that is being optimized
        def loss_fn(ro_model, control_gates):
            padded_control_gates = controll_padding(
                control_gates,
                params.startN,
                params.stopN,
                params.N,
            )
            phis = ro_sim.compute_quantum_channels(ro_model, padded_control_gates, fast_jit=params.fast_jit)
            return -mutual_information(phis[-1])

        control_gates, learning_curve = optimize(
            loss_fn,
            ro_model,
            control_gates,
            params.number_of_epoches,
            params.epoch_size,
            params.learning_rate,
        )

        # LOGGING SIMULATED DATA
        save_data(control_gates, dir_path + '/control_gates.pickle')
        save_data(learning_curve, dir_path + '/learning_curve.pickle')
        
        print("The optimal control sequance for subexperiment #{} is found.".format(i+1))
        
        control_gates = controll_padding(
            control_gates,
            params.startN,
            params.stopN,
            params.N,
        )

        # EXACT DYNAMICS SIMULATION WITH CONTROL
        controlled_quantum_channels = ex_sim.compute_quantum_channels(
            ex_sim_state,
            ex_env_state,
            gates_layer,
            control_gates,
        )
        controlled_mutual_information = mutual_information(controlled_quantum_channels)
        controlled_exact_density_matrices = channels2rhos(controlled_quantum_channels, system_state)
        
        # LOGGING SIMULATED DATA
        save_data(controlled_quantum_channels, dir_path + '/controlled_quantum_channels.pickle')
        save_data(controlled_exact_density_matrices, dir_path + '/controlled_exact_density_matrices.pickle')
        save_data(controlled_mutual_information, dir_path + '/controlled_mutual_information.pickle')

        # REDUCED_ORDER MODEL BASED DYNAMICS SIMULATION WITH CONTROL
        controlled_ro_model_based_density_matrices = ro_sim.compute_dynamics(ro_model, control_gates, system_state, fast_jit=params.fast_jit)
        
        # LOGGING SIMULATED DATA
        save_data(controlled_ro_model_based_density_matrices, dir_path + '/controlled_ro_based_density_matrices.pickle')
        
        print("Exact dynamics simulation under optimal control sequence for subexperiment #{} is done.".format(i+1))

        #SIMPLE PLOTTING
        
        plt.figure()
        plt.plot(ro_dim_vs_time)
        plt.ylabel('Environment dimension')
        plt.xlabel('N')
        plt.yscale('log')
        plt.savefig(dir_path + '/env_dimension.pdf')

        plt.figure()
        plt.plot(controlled_mutual_information[:, params.system_qubit], 'r')
        plt.plot(zero_control_mutual_information[:, params.system_qubit], 'b')
        plt.legend(['controlled mutual information', 'zero controll mutual information'])
        plt.ylabel('Amplitude')
        plt.xlabel('N')
        plt.savefig(dir_path + '/mutual_information_dynamics.pdf')

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
        i += 1

run_experiment(set_of_params)
