import numpy as np

def is_in_domain(X, domain):
    return ((np.min(X) < domain[0]) or (np.max(X) > domain[1]))


def make_local_states(domain, n_states):
    return np.linspace(domain[0],domain[1], max(n_states) + 1)


def discretize(X, domain, n_states):
    H = make_local_states(domain, n_states)    
    X_ = np.maximum(1-(abs(X[..., None]-H))/(H[1]-H[0]),0)
    return X_[..., list(n_states)]


def primitive_basis(X, domain=[-1,1], n_states=2):
    if not is_in_domain(X, domain):
        if isinstance(n_states, int):
            n_states = np.arange(n_states)
        return discretize(X, domain, n_states)
    else:
        raise RuntimeError("X must be within the specified domain")
