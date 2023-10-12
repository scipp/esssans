# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Loading and masking specific to the ISIS Sans2d instrument and files stored in Scipp's
HDF5 format.
"""
from typing import Optional

import scipp as sc
import scippneutron as scn

from .common import gravity_vector
from .types import (
    DirectBeamNumberOfSamplingPoints,
    DirectBeamSamplingWavelengthWidth,
    DirectBeamWavelengthSamplingPoints,
    Filename,
    NeXusMonitorName,
    MonitorType,
    RawData,
    RawMonitor,
    RunType,
    SampleRun,
    SourcePosition,
    WavelengthBins,
)


def define_wavelength_sampling_points(
    wavelength_bins: WavelengthBins,
    sampling_wavelength_width: DirectBeamSamplingWavelengthWidth,
    nsamples: DirectBeamNumberOfSamplingPoints,
) -> DirectBeamWavelengthSamplingPoints:
    sampling_half_width = sampling_wavelength_width / 2
    return sc.linspace(
        dim='wavelength',
        start=wavelength_bins[0] + sampling_half_width,
        stop=wavelength_bins[-1] - sampling_half_width,
        num=nsamples,
    )


providers = [
    define_wavelength_sampling_points,
]
"""
Providers for direct beam
"""
