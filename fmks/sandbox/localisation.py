from scipy.linalg import lstsq
import numpy as np

from .fext import curry, pipe, array_from_tuple, fmap
from .fext import fftshift, rfftn, irfftn


@curry
def _lstsq_slice(fx_data, fy_data, redundancy_func, ijk):
    return (ijk + redundancy_func(ijk),
            lstsq(fx_data[(slice(None),) + ijk + redundancy_func(ijk)],
                  fy_data[(slice(None),) + ijk],
                  np.finfo(float).eps * 1e4)[0])


def _fit_fourier(fx_data, fy_data, redundancy_func):
    lstsq_ijk = _lstsq_slice(fx_data, fy_data, redundancy_func)
    return pipe(
        fmap(lstsq_ijk, np.ndindex(fx_data.shape[1:-1])),
        list,
        array_from_tuple(shape=fx_data.shape[1:], dtype=np.complex)
    )


def _faxes(arr):
    return np.arange(arr.ndim - 2) + 1


@curry
def _fit_disc(y_data, x_data, redundancy_func):
    return pipe(
        [x_data, y_data],
        fmap(rfftn(axes=_faxes(x_data))),
        lambda x: _fit_fourier(*x, redundancy_func)
    )


@curry
def fit(x_data, y_data, basis):
    return _fit_disc(y_data, *basis(x_data))


@curry
def _predict_disc(x_data, coeff):
    return pipe(
        rfftn(x_data, axes=_faxes(x_data)),
        lambda x: np.sum(x * coeff[None], axis=-1),
        irfftn(axes=_faxes(x_data), s=x_data.shape[1:-1])
    ).real


@curry
def predict(x_data, coeff, basis):
    return _predict_disc(basis(x_data)[0], coeff)


def _ini_axes(arr):
    return np.arange(arr.ndim - 1)


@curry
def coeff_to_real(coeff, new_shape):
    return pipe(
        coeff,
        irfftn(axes=_ini_axes(coeff), s=new_shape),
        fftshift(axes=_ini_axes(coeff))
    )
