import chex
from dataclasses import dataclass, astuple
from typing import Tuple, Callable


@chex.dataclass
class ROMDynamicsGenerator:
    """These three arrays describe evolution operator for reduced state."""

    ker_top: chex.ArrayDevice
    ker_mid: chex.ArrayDevice
    ker_bottom: chex.ArrayDevice


@dataclass(frozen=True)
class ExactSimulatorState:
    """This is the state of the exact dynamics simulator."""

    number_of_qubits: int
    system_qubit_number: int
    controlled_qubit_number: int
    discrete_time: int

@dataclass(frozen=True)
class ExperimentParameters:
    """This dataclass keeps all the parameters of a possible experiment."""

    N: Tuple[int]
    n: Tuple[int]
    tau: Tuple[float]
    hx: Tuple[float]
    hy: Tuple[float]
    hz: Tuple[float]
    Jx: Tuple[float]
    Jy: Tuple[float]
    Jz: Tuple[float]
    system_qubit: Tuple[int]
    source_qubit: Tuple[int]
    system_state: Tuple[int]
    env_single_spin_state: Tuple[int]
    eps: Tuple[float]
    startN: Tuple[int]
    stopN: Tuple[int]
    full_truncation: Tuple[bool]
    truncate_when: Tuple[int]
    random_seed: Tuple[int]
    learning_rate: Tuple[float]
    epoch_size: Tuple[int]
    number_of_epoches: Tuple[int]
    fast_jit: Tuple[bool]

    def __iter__(self):
        return iter(astuple(self))  
