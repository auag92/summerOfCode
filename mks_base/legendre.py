import numpy as np
import numpy.polynomial.legendre as leg


def is_in_domain(X, domain):
    return ((np.min(X) < domain[0]) or (np.max(X) > domain[1]))


def scale_x(X, domain):
    return ((2.*X-domain[0]-domain[1])/(domain[1]-domain[0]))


def norm(n_states):
    return (2.*np.array(n_states)+1)/2.


def coeff(n_states):
    return np.eye(len(n_states))*norm(n_states)


def leg_x(X, domain, n_states):
     return (leg.legval(scale_x(X, domain),coeff(n_states)))


def discretize(X, domain=[-1,1], n_states=np.arange(2)):
        return rollaxis_(leg_x(X, domain, n_states))


def rollaxis_(X):
    return np.rollaxis(X, 0, len(X.shape))


def legendre_basis(X, domain=[-1,1], n_states=2):
    if not is_in_domain(X, domain):
        if isinstance(n_states, int):
            return discretize(X, domain, np.arange(n_states))
    else:
        raise RuntimeError("X must be within the specified domain")
