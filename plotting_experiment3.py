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

def latex(x):
    return list(map(lambda y: r'${}$'.format(y), x))

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

        plt.figure(figsize=(7, 2))
        
        # plotting mutual information dynamics under optimal control
        ax = plt.subplot(2, 2, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False) 
        #plt.ylim(bottom=params['n'], top=-1)
        plt.title(r'$J_z=h_x={}$, $ \Delta N = {}$'.format(params['hx'], params['stopN']-params['startN']))
        plt.yticks([0, params['n']-1], latex([1, params['n']]))
        plt.xticks([], [])
        plt.imshow(jnp.log(1e-2 + controlled_mutual_information.T), cmap='Spectral', interpolation='none', aspect='auto')
        plt.axvspan(params['startN'], params['stopN'], color='blue', alpha=0.2)

        # plotting mutual information dynamics without external control
        ax = plt.subplot(2, 2, 3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        #plt.ylim(bottom=params['n'], top=-1)
        plt.yticks([0, params['n']-1], latex([1, params['n']]))
        plt.imshow(jnp.log(1e-2 + zero_control_mutual_information.T), cmap='Spectral', interpolation='none', aspect='auto')

        # plotting comparision of mutual information dynamics for controled/wo control cases
        plt.subplot(1, 2, 2)
        plt.plot(controlled_mutual_information[:, system_qubit], 'r')
        plt.plot(zero_control_mutual_information[:, system_qubit], 'b')
        plt.axvspan(params['startN'], params['stopN'], color='blue', alpha=0.2)
        plt.legend([r'$I^{\rm opt}$', r'$I$'],
                fontsize=14,
                framealpha=1)
        plt.savefig('./plots_experiment3/mutual_information_dynamics_comparison_{}_{}_{}.svg'.format(params['hx'], params['system_qubit'], params['stopN'] - params['startN']))
