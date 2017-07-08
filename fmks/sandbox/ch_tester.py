import numpy as np
import math as math
import os
import numpy.fft as fftmodule
import pymks

def initField(size=(101, 101), n_samples=1, factor=1e-2):
    X0 = (2 * np.random.random((n_samples,) + size) - 1)*factor
    return X0

def ch_solver(X, dx=0.25, dt=0.001, gamma = 1.):

    N = X.shape[1]
    if not np.all(np.array(X.shape[1:]) == N):
        raise RuntimeError("X must represent a square domain")

    L = dx * N
    k = np.arange(N)

    N1 = math.floor(N/2)
    N2 = math.ceil(N/2)

    k[N2:] = (k - N1)[:N1]
    k = k * 2 * np.pi / L

    i_ = np.indices(X.shape[1:])
    ksq = np.sum(k[i_] ** 2, axis=0)[None]

    axes = np.arange(len(X.shape) - 1) + 1
    FX  = fftmodule.fftn(X, axes=axes)
    FX3 = fftmodule.ifftn(X, axes=axes)

    a1 = 3.
    a2 = 0.

    explicit = ksq * (a1 - gamma * a2 * ksq)
    implicit = ksq * ((1 - a1) - gamma * (1 - a2) * ksq)

    Fy = (FX * (1 + dt * explicit) - ksq * dt * FX3) / (1 - dt * implicit)
    response = fftmodule.ifftn(Fy,axes=axes).real
    return response

def main():
    directory = "data/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    X = initField(n_samples=400)
    filename = "data/%s_%d_%d" % ('phi', 101, 0)
    np.save(filename, X)
    tsteps = 5000

    for i in range(tsteps):
        print(i)
        X = ch_solver(X, dx=0.25, dt=0.01, gamma = 1.)
        if ((i+1)%100) == 0:
            filename = "data/%s_%d_%d" % ('phi', 101, i)
            np.save(filename, X)

if __name__ == "__main__":
    main()
