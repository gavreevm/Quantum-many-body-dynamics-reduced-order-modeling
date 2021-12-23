import jax.numpy as jnp
from jax import lax
from typing import Tuple
import chex


# In this file we implement Riemannian AMSGrad optimizer that perform optimization over
# isometric matrices (set of unitray matrices is a special case)


@chex.dataclass
class OptimizerState:
    learning_rate: float
    beta1: float
    beta2: float
    eps: float
    m: chex.ArrayDevice
    v: chex.ArrayDevice
    v_hat: chex.ArrayDevice
    iter: float


def _adj(a: jnp.ndarray) -> jnp.ndarray:
    """[This function returns set of adjoint matrices.]

    Args:
        a (complex valued jnp.ndarray of shape (..., n1, n2)): [set of matrix]

    Returns:
        complex valued jnp.ndarray of shape (..., n2, n1): [set of adjoint matrices]
    """

    return jnp.swapaxes(a, -1, -2).conj()


def _inner(u: jnp.ndarray,
           vec1: jnp.ndarray,
           vec2: jnp.ndarray) -> jnp.ndarray:
    """[This function returns manifold wise inner product of two vectors
    from a tangent space of a point u from a product of complex
    Stiefel manifolds.]

    Args:
        u (complex valued jnp.ndarray of shape (..., n, m)): [point]
        vec1 (complex valued jnp.ndarray of shape (..., n, m)): [first tangent vector]
        vec2 (complex valued jnp.ndarray of shape (..., n, m)): [second tangent vector] 

    Returns:
        jnp.ndarray: [description]
    """

    s_sq = (vec1.conj() * vec2).sum(keepdims=True, axis=(-2, -1))
    return jnp.real(s_sq).astype(dtype=u.dtype)


def _egrad_to_rgrad(u: jnp.ndarray,
                    egrad: jnp.ndarray) -> jnp.ndarray:
    """[This function returns Riemannian gradient from
    Eucledian one.]

    Args:
        u (complex valued jnp.ndarray of shape (..., n, m)): [point]
        egrad (complex valued jnp.ndarray of shape (..., n, m)): [Eucledian gradient]

    Returns:
        complex valued jnp.ndarray of shape (..., n, m): [Riemannian gradient]
    """

    return (0.5 * u @ (_adj(u) @ egrad - _adj(egrad) @ u)
    + egrad
    - u @ (_adj(u) @ egrad))


def _proj(u: jnp.ndarray, vec: jnp.ndarray) -> jnp.ndarray:
    """[This function returns projecyion of a vector on the tangent space.]

    Args:
        u (complex valued jnp.ndarray of shape (..., n, m)): [point]
        vec (complex valued jnp.ndarray of shape (..., n, m)): [vector]

    Returns:
        complex valued jnp.ndarray of shape (..., n, m): [the projection on the tangent space of the vector]
    """

    return 0.5 * u @ (_adj(u) @ vec - _adj(vec) @ u) + vec - u @ (_adj(u) @ vec)


def _retraction(u: jnp.ndarray, vec: jnp.ndarray) -> jnp.ndarray:
    """[This function performs retraction (first order approximation of the exponential map).]

    Args:
        u (complex valued jnp.ndarray of shape (..., n, m)): [point]
        vec (complex valued jnp.ndarray of shape (..., n, m)): [vector pointing retraction direction]

    Returns:
        complex valued jnp.ndarray of shape (..., n, m): [new point]
    """

    new_u = u + vec
    v, _, w = jnp.linalg.svd(new_u, full_matrices=False)
    return v @ w


def _retraction_transposrt(u: jnp.ndarray, vec1: jnp.ndarray, vec2: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """[This function performs retraction and vector transport.]

    Args:
        u (complex valued jnp.ndarray of shape (..., n, m)): [point]
        vec1 (complex valued jnp.ndarray of shape (..., n, m)): [vector that is being transported]
        vec2 (jnp.ndarray): [vector pointing retraction dnd vector tranposrt direction]

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: [transported vector and transported point]
    """

    new_u = _retraction(u, vec2)
    return new_u, _proj(new_u, vec1)


def init_optimizer_state(control_signal: jnp.ndarray,
                         learning_rate: float,
                         beta1: float = 0.9,
                         beta2: float = 0.999,
                         eps: float = 1e-8) -> OptimizerState:
    """[This function initializes the state of Riemannina AMSGrad optimizer.]

    Args:
        control_signal (complex valued jnp.ndarray of shape (..., n, m)): [description]
        learning_rate (float): [learning rate]
        beta1 (float, optional): [beta1 param]. Defaults to 0.9.
        beta2 (float, optional): [beta2 param]. Defaults to 0.999.
        eps (float, optional): [eps param]. Defaults to 1e-8.

    Returns:
        OptimizerState: [state of the optimizer]
    """

    state = OptimizerState(
        learning_rate = learning_rate,
        beta1 = beta1,
        beta2 = beta2,
        eps = eps,
        m = jnp.zeros_like(control_signal),
        v = jnp.zeros((*control_signal.shape[:-2], 1, 1), dtype=control_signal.dtype),
        v_hat = jnp.zeros((*control_signal.shape[:-2], 1, 1), dtype=control_signal.dtype),
        iter = 0, 
    )
    return state


def run_step(state: OptimizerState,
             grad: jnp.ndarray,
             control_signal: jnp.ndarray) -> Tuple[OptimizerState, jnp.ndarray]:
    """[This function performs Riemannian AMSGrad optimization step.]

    Args:
        state (OptimizerState): [optimizer state]
        grad (complex valued jnp.ndarray of shape (..., n, m)): [gradient]
        control_signal (complex valued jnp.ndarray of shape (..., n, m)): [current control signal]

    Returns:
        Tuple[OptimizerState, jnp.ndarray]: [new control signal and updated optimizer state]
    """

    rgrad = _egrad_to_rgrad(control_signal, grad.conj())
    state.m = state.beta1 * state.m + (1 - state.beta1) * rgrad
    state.v = state.beta2 * state.v + (1 - state.beta2) * _inner(control_signal, rgrad, rgrad)
    state.v_hat = lax.complex(jnp.maximum(jnp.real(state.v), jnp.real(state.v_hat)), jnp.imag(state.v))
    state.iter += 1

    learning_rate_corr = state.learning_rate * jnp.sqrt(1 - state.beta2 ** state.iter) / (1 - state.beta1 ** state.iter)
    search_dir = -learning_rate_corr * state.m / (jnp.sqrt(state.v_hat) + state.eps)
    control_signal, state.m = _retraction_transposrt(control_signal, state.m, search_dir)
    return control_signal, state
