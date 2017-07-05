import fmks
import numpy as np
import numpy.polynomial.legendre as leg

def is_in_domain(X, domain):
    return ((np.min(X) < domain[0]) or (np.max(X) > domain[1]))

def scaled_x(X, domain):
    return ((2.*X-domain[0]-domain[1])/(domain[1]-domain[0]))

def norm(n_states):
    return (2.*np.array(n_states)+1)/2.

def coeff(n_states):
    return np.eye(len(n_states))*norm(n_states)

def leg_x(X, domain, n_states):
     return (leg.legval(scaled_x(X, domain),coeff(n_states)))

def discretize(X, domain, n_states):
    if not is_in_domain(X, domain):
        if isinstance(n_states, int):
            n_states = np.arange(n_states)
        X_Legendre = leg_x(X, domain, n_states)
        return np.rollaxis(X_Legendre, 0, len(X_Legendre.shape))
    else:
        raise RuntimeError("X must be within the specified domain")
