import numpy as np

def is_in_domain(X, domain):
    return ((np.min(X) < domain[0]) or (np.max(X) > domain[1]))


def makeStates(domain, n_states):
    return np.linspace(domain[0], domain[1], max(n_states) + 1)


def discretize_(X, states):
    return np.maximum(1-(abs(X[..., None]-states))/(states[1]-states[0]),0)


def discretize(X, domain, n_states): 
    return discretize_(X, states=makeStates(domain, n_states))[..., list(n_states)]


def primitive_basis(X, domain=[-1,1], n_states=2):
    if not is_in_domain(X, domain):
        if isinstance(n_states, int):
            return discretize(X, domain, np.arange(n_states))
        else:
            return discretize(X, domain, n_states)
    else:
        raise RuntimeError("X must be within the specified domain")
