# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Callable, List

import numpy as np
import sciline
import scipp as sc
from loki_common import make_params

import esssans as sans
from esssans.types import DimsToKeep, QBins


def get_I0(q_loc: sc.Variable) -> sc.Variable:
    from esssans.loki.data import get_path

    data = np.loadtxt(get_path('PolyGauss_I0-50_Rg-60.txt'))
    qcoord = sc.array(dims=["Q"], values=data[:, 0], unit='1/angstrom')
    theory = sc.DataArray(
        data=sc.array(dims=["Q"], values=data[:, 1], unit=''), coords={"Q": qcoord}
    )
    ind = np.argmax((qcoord > q_loc).values)
    I0 = (theory.data[ind] - theory.data[ind - 1]) / (qcoord[ind] - qcoord[ind - 1]) * (
        q_loc - qcoord[ind - 1]
    ) + theory.data[ind - 1]
    return I0


def loki_providers() -> List[Callable]:
    return list(sans.providers + sans.loki.providers)


def test_can_compute_direct_beam_for_all_pixels():
    n_wavelength_bands = 10
    params = make_params(n_wavelength_bands=n_wavelength_bands)
    providers = loki_providers()
    pipeline = sciline.Pipeline(providers, params=params)
    I0 = get_I0(sc.midpoints(params[QBins])[0])

    results = sans.direct_beam(pipeline=pipeline, I0=I0, niter=4)
    # Unpack the final result
    iofq_full = results[-1]['iofq_full']
    iofq_slices = results[-1]['iofq_slices']
    direct_beam_function = results[-1]['direct_beam']
    assert iofq_full.dims == ('Q',)
    assert iofq_slices.dims == ('band', 'Q')
    assert iofq_slices.sizes['band'] == n_wavelength_bands
    assert direct_beam_function.sizes['wavelength'] == n_wavelength_bands


def test_can_compute_direct_beam_per_layer():
    n_wavelength_bands = 10
    params = make_params(n_wavelength_bands=n_wavelength_bands)
    params[DimsToKeep] = ['layer']
    providers = loki_providers()
    pipeline = sciline.Pipeline(providers, params=params)
    I0 = get_I0(sc.midpoints(params[QBins])[0])

    results = sans.direct_beam(pipeline=pipeline, I0=I0, niter=4)
    # Unpack the final result
    iofq_full = results[-1]['iofq_full']
    iofq_slices = results[-1]['iofq_slices']
    direct_beam_function = results[-1]['direct_beam']
    assert iofq_full.dims == (
        'layer',
        'Q',
    )
    assert iofq_slices.dims == ('band', 'layer', 'Q')
    assert iofq_slices.sizes['band'] == n_wavelength_bands
    assert iofq_slices.sizes['layer'] == 4
    assert direct_beam_function.sizes['wavelength'] == n_wavelength_bands
    assert direct_beam_function.sizes['layer'] == 4