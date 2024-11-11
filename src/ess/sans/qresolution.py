# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""Q-resolution calculation for SANS data."""

from typing import NewType

import scipp as sc
from scipp.constants import pi

from .common import mask_range
from .conversions import ElasticCoordTransformGraph
from .normalization import _reduce
from .types import (
    CleanQ,
    Denominator,
    DimsToKeep,
    ProcessedWavelengthBands,
    QBins,
    SampleRun,
    WavelengthMask,
)

DeltaR = NewType("DeltaR", sc.Variable)
"""Virtual ring width on the detector."""

SampleApertureRadius = NewType("SampleApertureRadius", sc.Variable)
"""Sample aperture radius, R2."""
SourceApertureRadius = NewType("SourceApertureRadius", sc.Variable)
"""Source aperture radius, R1."""
SigmaModerator = NewType("SigmaModerator", sc.DataArray)
"""Moderator time spread as a function of wavelength."""
CollimationLength = NewType("CollimationLength", sc.Variable)
"""Collimation length."""
ModeratorTimeSpread = NewType("ModeratorTimeSpread", sc.DataArray)
"""Moderator time-spread as a function of wavelength."""


QResolutionByPixel = NewType("QResolutionByPixel", sc.DataArray)
QResolutionByQ = NewType("QResolutionPixelTermGroupedQ", sc.DataArray)
QResolutionByWavelength = NewType("QResolutionByWavelength", sc.DataArray)
QResolution = NewType("QResolution", sc.DataArray)


def q_resolution_by_pixel(
    detector: CleanQ[SampleRun, Denominator],
    delta_r: DeltaR,
    sample_aperture: SampleApertureRadius,
    source_aperture: SourceApertureRadius,
    collimation_length: CollimationLength,
    moderator_time_spread: ModeratorTimeSpread,
    graph: ElasticCoordTransformGraph,
) -> QResolutionByPixel:
    """
    Calculate the Q-resolution per pixel.

    We compute this based on CleanQ[SampleRun, Denominator]. This ensures that

    1. We get the correct Q-value, based on wavelength binning used elsewhere.
    2. Masks are included.
    3. We do not depend on neutron data, by using the denominator instead of the
       numerator.
    """
    detector = detector.transform_coords("L2", graph=graph, keep_inputs=False)
    L2 = detector.coords["L2"]
    L3 = sc.reciprocal(sc.reciprocal(collimation_length) + sc.reciprocal(L2))
    result = detector.copy(deep=False)
    pixel_term = (
        3 * ((source_aperture / collimation_length) ** 2 + (sample_aperture / L3) ** 2)
        + (delta_r / L2) ** 2
    )
    inv_lambda2 = sc.reciprocal(detector.coords['wavelength'] ** 2)
    Q2 = detector.coords['Q'] ** 2
    result.data = (pi**2 / 3) * inv_lambda2 * pixel_term + Q2 * (
        moderator_time_spread**2 * inv_lambda2
    )
    return QResolutionByPixel(result)


def q_resolution_by_q(
    data: QResolutionByPixel, q_bins: QBins, dims_to_keep: DimsToKeep
) -> QResolutionByQ:
    dims = [dim for dim in data.dims if dim not in (*dims_to_keep, 'wavelength')]
    return QResolutionByQ(data.bin(Q=q_bins, dim=dims))


def mask_qresolution_in_wavelength(
    resolution: QResolutionByQ, mask: WavelengthMask
) -> QResolutionByWavelength:
    """
    Compute the masked Q-resolution in (Q, lambda) space.

    CleanSummedQ has been summed over pixels but not over wavelengths. This is exactly
    what is required for performing the remaining scaling and addition of the moderator
    term to obtain the Q-resolution. The result is still in (Q, lambda) space.
    """
    if mask is not None:
        resolution = mask_range(resolution, mask=mask)
    return QResolutionByWavelength(resolution)


def reduce_resolution_q(
    data: QResolutionByWavelength, bands: ProcessedWavelengthBands
) -> QResolution:
    return QResolution(_reduce(data, bands=bands))


providers = (
    q_resolution_by_pixel,
    q_resolution_by_q,
    mask_qresolution_in_wavelength,
    reduce_resolution_q,
)
