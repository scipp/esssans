# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc

from .types import (
    CorrectForGravity,
    Incident,
    NeXusMonitorName,
    NonBackgroundWavelengthRange,
    QBins,
    Transmission,
    UncertaintyBroadcastMode,
    WavelengthBands,
    WavelengthBins,
    WavelengthMask,
)


def sans2d() -> dict:
    params = {}
    params[NeXusMonitorName[Incident]] = 'monitor2'
    params[NeXusMonitorName[Transmission]] = 'monitor4'
    # Is this always the same?
    mask_interval = sc.array(dims=['wavelength'], values=[2.21, 2.59], unit='angstrom')
    params[WavelengthMask] = sc.DataArray(
        sc.array(dims=['wavelength'], values=[True]),
        coords={'wavelength': mask_interval},
    )
    # Is this always the same?
    params[NonBackgroundWavelengthRange] = sc.array(
        dims=['wavelength'], values=[0.7, 17.1], unit='angstrom'
    )
    return params


def base_params() -> dict:
    params = {}
    params[CorrectForGravity] = False
    params[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.upper_bound
    return params


def setup(*, wavelength_bins: sc.Variable, q_bins: sc.Variable) -> dict:
    params = base_params()
    params[WavelengthBands] = WavelengthBands(
        sc.concat([wavelength_bins[0], wavelength_bins[-1]], wavelength_bins.dim)
    )
    params[WavelengthBins] = WavelengthBins(wavelength_bins)
    params[QBins] = QBins(q_bins)
    return params
