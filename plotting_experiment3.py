import os
#os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\bin")
import jax
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

from simulators.logger import load_params, load_data
from experiments_utils import rho2bloch

data = {}
for i, (root, dirs, files) in enumerate(os.walk("./experiment3_data")):
    if i > 0:
        params = load_params(os.path.join(root, 'params.txt'))
        initial_state = jnp.array(params['system_state'])
        initial_state = jnp.tensordot(initial_state, initial_state.conj(), axes=0)
        initial_state = initial_state[jnp.newaxis]
        system_qubit = params['system_qubit']
        first_row = 2 * jnp.log(2) * jax.nn.one_hot(jnp.array(system_qubit), params['n'])[jnp.newaxis]
        controlled_mutual_information = jnp.concatenate([first_row, load_data(os.path.join(root, 'controlled_mutual_information.pickle'))], axis=0)
        zero_control_mutual_information = jnp.concatenate([first_row, load_data(os.path.join(root, 'zero_control_mutual_information.pickle'))], axis=0)

        # plotting mutual information dynamics under optimal control
        plt.figure()
        plt.imshow(controlled_mutual_information.T, cmap='Spectral')
        plt.axvspan(params['startN'], params['stopN'], color='blue', alpha=0.2)
        plt.savefig('./plots_experiment3/controlled_mutual_information_{}_{}_{}.pdf'.format(params['n'], params['system_qubit'], params['Jz']))

        # plotting mutual information dynamics without external control
        plt.figure()
        plt.imshow(zero_control_mutual_information.T, cmap='Spectral')
        plt.savefig('./plots_experiment3/zero_control_mutual_information_{}_{}_{}.pdf'.format(params['n'], params['system_qubit'], params['Jz']))

        # plotting comparision of mutual information dynamics for controled/wo control cases
        plt.figure()
        plt.plot(controlled_mutual_information[:, system_qubit], 'r')
        plt.plot(zero_control_mutual_information[:, system_qubit], 'b')
        plt.axvspan(params['startN'], params['stopN'], color='blue', alpha=0.2)
        plt.legend([r'$I^{\rm opt}_{1\rightarrow 1}$', r'$I_{1\rightarrow 1}$'],
                   fontsize=16,
                   frameon=False)
        plt.savefig('./plots_experiment3/mutual_information_dynamics_comparison_{}_{}_{}.pdf'.format(params['n'], params['system_qubit'], params['Jz']))
