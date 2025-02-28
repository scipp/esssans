# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import sys
from pathlib import Path

import scipp as sc
from scipp.scipy.interpolate import interp1d

from ess import loki, sans
from ess.sans.types import (
    BeamCenter,
    DimsToKeep,
    QBins,
    WavelengthBands,
    WavelengthBins,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import make_workflow


def _get_I0(qbins: sc.Variable) -> sc.Variable:
    Iq_theory = sc.io.load_hdf5(loki.data.loki_tutorial_poly_gauss_I0())
    f = interp1d(Iq_theory, 'Q')
    return f(sc.midpoints(qbins)).data[0]


def test_can_compute_direct_beam_for_all_pixels():
    n_wavelength_bands = 10
    pipeline = make_workflow()
    edges = pipeline.compute(WavelengthBins)
    pipeline[WavelengthBands] = sc.linspace(
        'wavelength', edges.min(), edges.max(), n_wavelength_bands + 1
    )
    pipeline[BeamCenter] = sc.vector([0, 0, 0], unit='m')
    I0 = _get_I0(qbins=pipeline.compute(QBins))

    results = sans.direct_beam(workflow=pipeline, I0=I0, niter=4)
    iofq_full = results[-1]['iofq_full']
    iofq_bands = results[-1]['iofq_bands']
    direct_beam_function = results[-1]['direct_beam']

    assert iofq_full.dims == ('Q',)
    assert iofq_bands.dims == ('band', 'Q')
    assert iofq_bands.sizes['band'] == n_wavelength_bands
    assert direct_beam_function.sizes['wavelength'] == n_wavelength_bands


def test_can_compute_direct_beam_with_overlapping_wavelength_bands():
    n_wavelength_bands = 10
    # Bands have double the width
    pipeline = make_workflow()
    edges = pipeline.compute(WavelengthBins)
    edges = sc.linspace('band', edges.min(), edges.max(), n_wavelength_bands + 2)
    pipeline[WavelengthBands] = sc.concat(
        [edges[:-2], edges[2::]], dim='wavelength'
    ).transpose()

    pipeline[BeamCenter] = sc.vector([0, 0, 0], unit='m')
    I0 = _get_I0(qbins=pipeline.compute(QBins))

    results = sans.direct_beam(workflow=pipeline, I0=I0, niter=4)
    iofq_full = results[-1]['iofq_full']
    iofq_bands = results[-1]['iofq_bands']
    direct_beam_function = results[-1]['direct_beam']

    assert iofq_full.dims == ('Q',)
    assert iofq_bands.dims == ('band', 'Q')
    assert iofq_bands.sizes['band'] == n_wavelength_bands
    assert direct_beam_function.sizes['wavelength'] == n_wavelength_bands


def test_can_compute_direct_beam_per_layer():
    n_wavelength_bands = 10
    pipeline = make_workflow()
    edges = pipeline.compute(WavelengthBins)
    pipeline[WavelengthBands] = sc.linspace(
        'wavelength', edges.min(), edges.max(), n_wavelength_bands + 1
    )
    pipeline[DimsToKeep] = ['layer']
    pipeline[BeamCenter] = sc.vector([0, 0, 0], unit='m')
    I0 = _get_I0(qbins=pipeline.compute(QBins))

    results = sans.direct_beam(workflow=pipeline, I0=I0, niter=4)
    iofq_full = results[-1]['iofq_full']
    iofq_bands = results[-1]['iofq_bands']
    direct_beam_function = results[-1]['direct_beam']

    assert iofq_full.dims == ('layer', 'Q')
    assert iofq_bands.dims == ('band', 'layer', 'Q')
    assert iofq_bands.sizes['band'] == n_wavelength_bands
    assert iofq_bands.sizes['layer'] == 4
    assert direct_beam_function.sizes['wavelength'] == n_wavelength_bands
    assert direct_beam_function.sizes['layer'] == 4


def test_can_compute_direct_beam_per_layer_and_straw():
    n_wavelength_bands = 10
    # The test fails when using small files because the counts are too low, leading to
    # divisions by zero and NaNs in the result.
    pipeline = make_workflow(use_small_files=False)
    edges = pipeline.compute(WavelengthBins)
    pipeline[WavelengthBands] = sc.linspace(
        'wavelength', edges.min(), edges.max(), n_wavelength_bands + 1
    )
    pipeline[DimsToKeep] = ('layer', 'straw')
    pipeline[BeamCenter] = sc.vector([0, 0, 0], unit='m')
    I0 = _get_I0(qbins=pipeline.compute(QBins))

    results = sans.direct_beam(workflow=pipeline, I0=I0, niter=4)
    iofq_full = results[-1]['iofq_full']
    iofq_bands = results[-1]['iofq_bands']
    direct_beam_function = results[-1]['direct_beam']

    assert iofq_full.dims == ('layer', 'straw', 'Q')
    assert iofq_bands.dims == ('band', 'layer', 'straw', 'Q')
    assert iofq_bands.sizes['band'] == n_wavelength_bands
    assert iofq_bands.sizes['layer'] == 4
    assert iofq_bands.sizes['straw'] == 7
    assert direct_beam_function.sizes['wavelength'] == n_wavelength_bands
    assert direct_beam_function.sizes['layer'] == 4
    assert direct_beam_function.sizes['straw'] == 7
