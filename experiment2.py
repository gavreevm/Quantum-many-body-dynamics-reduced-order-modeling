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
from simulators.models_utils import params2gates_layer
from simulators.exact_simulator import ExactSimulator
from simulators.reduced_order_simulator import ReducedOrderSimulator
from simulators.control import optimize
from simulators.dataclasses import ExperimentParameters
from simulators.exact_simulator_utils import complete_system
from experiments_utils import mutual_information, rom2exact_init_state_converter, zero_control_seq, random_control_seq, channels2rhos, rho2bloch


# This file performs simulation of the system using both the exact
# simulator and the reduced-order simulator without control signal
# and with the optimal control.

# Number of the source qubit must be greater than number of the system qubit!

# ------------------ Description of model parameters ------------------------ #
# N --------------------- number of discrete time steps
# n --------------------- number of spins
# tau ------------------- time step duration
# hi -------------------- i-th component of the external magnetic field (i \in {x, y, z})
# Ji -------------------- i-th component of the coupling (i \in {x, y, z})
# system_qubit ---------- the number of the system spin (ranges from 0, to n-1)
# source_qubit ---------- the number of qubit whose state one needs to reconstruct
# system_state ---------- psi vector of the system spin
# env_single_spin_state - psi vector of each spin in the environment
# eps ------------------- accuracy of truncation
# full_truncation ------- flag showing whether to use full truncation or not
# truncate_when --------- the bond dim. value threshold
# random_seed ----------- random seed
# learning_rate --------- learning rate that is used while searching of the optimal control sequance
# epoch_size ------------ optimization epoch size (number of iteration per epoch)
# number_of_epoches ----- total number of optimization epoches
# fast_jit -------------- ON/OFF fast_jit


# experiment parameters
set_of_params = ExperimentParameters(
    N = 3 * (50,),
    n = (15, 21, 21),
    tau = 3 * (0.15,),
    hx = 3 * (0.2,),
    hy = 3 * (0.2,),
    hz = 3 * (0.2,),
    Jx = 3 * (0.9,),
    Jy = 3 * (1.,),
    Jz = 3 * (1.1,),
    system_qubit = (0, 0, 2),
    source_qubit = (14, 20, 18),
    system_state = 3 * ([1, 0],),  # this parameter is not active in this experiment
    env_single_spin_state = 3 * ([0, 1],),
    eps = 3 * (1e-2,),
    startN = 3 * (None,),  # this parameter is not active in this experiment
    stopN = 3 * (None,),  # this parameter is not active in this experiment
    full_truncation = 3 * (True,),
    truncate_when = 3 * (512,),
    random_seed = 3 * (42,),
    learning_rate = 3 * (0.03,),
    epoch_size = 3 * (100,),
    number_of_epoches = 3 * (100,),
    fast_jit = 3 * (False,),
)


def run_experiment(set_of_params: ExperimentParameters):
    i = 0
    for params in zip(*set_of_params):
        params = ExperimentParameters(*params)

        print("Subexperiment #{} is run. \n".format(i+1))

        key = random.PRNGKey(params.random_seed)
        experiment_id = str(uuid.uuid4())
        dir_path = 'experiment2_data/' + experiment_id
        os.mkdir(dir_path)

        save_params(asdict(params), dir_path + '/params.txt')

        # gates
        gates_layer = params2gates_layer(params)
        save_data(gates_layer, dir_path + '/gates_layer.pickle')

        # initial env. states for the reduced-order simulator
        ro_env_state0 = (params.source_qubit - 1) * [jnp.array(params.env_single_spin_state, dtype=jnp.complex64)] +\
            [complete_system[0]] + (params.n - params.source_qubit - 1) * [jnp.array(params.env_single_spin_state, dtype=jnp.complex64)]
        ro_env_state1 = (params.source_qubit - 1) * [jnp.array(params.env_single_spin_state, dtype=jnp.complex64)] +\
            [complete_system[1]] + (params.n - params.source_qubit - 1) * [jnp.array(params.env_single_spin_state, dtype=jnp.complex64)]
        ro_env_state2 = (params.source_qubit - 1) * [jnp.array(params.env_single_spin_state, dtype=jnp.complex64)] +\
            [complete_system[2]] + (params.n - params.source_qubit - 1) * [jnp.array(params.env_single_spin_state, dtype=jnp.complex64)]
        ro_env_state3 = (params.source_qubit - 1) * [jnp.array(params.env_single_spin_state, dtype=jnp.complex64)] +\
            [complete_system[3]] + (params.n - params.source_qubit - 1) * [jnp.array(params.env_single_spin_state, dtype=jnp.complex64)]

        # system state
        system_state = jnp.array(params.env_single_spin_state, dtype=jnp.complex64)

        # initial env. state for the exact simulator
        ex_env_state = rom2exact_init_state_converter((params.n - 1) * [system_state])

        # initial non-zero control seq.
        control_gates = random_control_seq(key, params.N)

        # REDUCED_ORDER MODELS BUILDING
        ro_sim = ReducedOrderSimulator()
        ro_model0 = ro_sim.build_reduced_order_model(
            params.system_qubit,
            params.system_qubit,
            params.N,
            ro_env_state0,
            gates_layer,
            params.full_truncation,
            params.truncate_when,
            params.eps)
        ro_model1 = ro_sim.build_reduced_order_model(
            params.system_qubit,
            params.system_qubit,
            params.N,
            ro_env_state1,
            gates_layer,
            params.full_truncation,
            params.truncate_when,
            params.eps)
        ro_model2 = ro_sim.build_reduced_order_model(
            params.system_qubit,
            params.system_qubit,
            params.N,
            ro_env_state2,
            gates_layer,
            params.full_truncation,
            params.truncate_when,
            params.eps)
        ro_model3 = ro_sim.build_reduced_order_model(
            params.system_qubit,
            params.system_qubit,
            params.N,
            ro_env_state3,
            gates_layer,
            params.full_truncation,
            params.truncate_when,
            params.eps)
        
        ro_models = [
            ro_model0,
            ro_model1,
            ro_model2,
            ro_model3
        ]
        save_data(ro_models, dir_path + '/ro_models.pickle')
        
        # LOGGING REDUCED ORDER MODELS
        save_data(ro_models, dir_path + '/ro_models.pickle')
        
        print("Reduced-order models for subexperiment #{} are built.".format(i+1))
        print("Control sequance optimization is run:")

        # CONTROL SIGNAL OPTIMIZATION
        if params.fast_jit:
            ro_models = [ro_sim.preprocess_reduced_order_model(ro_model) for ro_model in ro_models]
            
        # This is the loss function that is being optimized
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
            #  Frobenius distance minimization
            loss_val = (
                jnp.linalg.norm(rho0 - rho_inp0) ** 2 +
                jnp.linalg.norm(rho1 - rho_inp1) ** 2 +
                jnp.linalg.norm(rho2 - rho_inp2) ** 2 +
                jnp.linalg.norm(rho3 - rho_inp3) ** 2
            )
            return loss_val

        control_gates, learning_curve = optimize(
            loss_fn,
            ro_models,
            control_gates,
            params.number_of_epoches,
            params.epoch_size,
            params.learning_rate,
        )

        # LOGGING SIMULATED DATA
        save_data(control_gates, dir_path + '/control_gates.pickle')
        save_data(learning_curve, dir_path + '/learning_curve.pickle')
        
        print("The optimal control sequance for subexperiment #{} is found.".format(i+1))

        # EXACT DYNAMICS SIMULATION WITH CONTROL
        ex_sim = ExactSimulator()
        ex_sim_state = ex_sim.initialize(
            params.n,
            params.source_qubit,
            params.system_qubit,
            params.N,
        )
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
        
        print("Exact dynamics simulation under optimal control sequence for subexperiment #{} is done.".format(i+1))


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
