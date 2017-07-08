from typing import Callable, Tuple
import numpy as np
from .fext import curry

def _discretize(x_data: np.ndarray, states: np.ndarray):
    return 1 - (abs(x_data[..., None] - states)) / (states[1] - states[0])


def _minmax(data, min_, max_):
    return np.minimum(np.maximum(data, min_), max_)


@curry
def discretize(x_data: np.ndarray,
               n_state: int,
               min_: float = 0.0,
               max_: float = 1.0):
    return np.maximum(
        _discretize(_minmax(x_data, min_, max_),
                    np.linspace(min_, max_, n_state)),
        0
    )


def redundancy(ijk: tuple) -> tuple:
    if np.all(np.array(ijk) == 0):
        return (slice(None),)
    return (slice(-1),)


@curry
def primitive_basis(x_data: np.ndarray,
                    n_state: int,
                    min_: float = 0.0,
                    max_: float = 1.0):
    return (discretize(x_data, n_state, min_=min_, max_=max_),
            redundancy)
