# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""Q-resolution calculation for SANS data."""

from typing import NewType

import scipp as sc
from scipp.constants import pi

from .common import mask_range
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


QResolutionPixelTerm = NewType("QResolutionPixelTerm", sc.DataArray)
QResolutionPixelTermGroupedQ = NewType("QResolutionPixelTermGroupedQ", sc.DataArray)
QResolutionByWavelength = NewType("QResolutionByWavelength", sc.DataArray)
QResolution = NewType("QResolution", sc.DataArray)


def pixel_term(
    detector: CleanQ[SampleRun, Denominator],
    delta_r: DeltaR,
    sample_aperture: SampleApertureRadius,
    source_aperture: SourceApertureRadius,
    collimation_length: CollimationLength,
) -> QResolutionPixelTerm:
    """
    Calculate the pixel term for Q-resolution.

    We compute this based on CleanQ[SampleRun, Denominator]. This ensures that

    1. We get the correct Q-value, based on wavelength binning used elsewhere.
    2. Masks are included.
    3. We do not depend on neutron data, by using the denominator instead of the
       numerator.
    """
    L2 = detector.coords["L2"]
    L3 = sc.reciprocal(sc.reciprocal(collimation_length) + sc.reciprocal(L2))
    result = detector.copy(deep=False)
    result.data = (
        3 * ((source_aperture / collimation_length) ** 2 + (sample_aperture / L3) ** 2)
        + (delta_r / L2) ** 2
    )
    # TODO
    # Give different name, as we do not actually feed into bin_in_q
    return QResolutionPixelTerm(result)


# TODO What is the point of naming the input CleanQ and output
# CleanSummedQ? It is not sharing functions, so use a different name
def groupby_q_max(
    data: QResolutionPixelTerm,
    q_bins: QBins,
    dims_to_keep: DimsToKeep,
) -> QResolutionPixelTermGroupedQ:
    # TODO Handle multi dim and dims_to_keep!
    # Can we use common helper function from bin_in_q?
    out = data.groupby('Q', bins=q_bins).max('detector_number')
    return QResolutionPixelTermGroupedQ(out)


def mask_and_compute_resolution_q(
    pixel_term: QResolutionPixelTermGroupedQ,
    mask: WavelengthMask,
    moderator_time_spread: ModeratorTimeSpread,
) -> QResolutionByWavelength:
    """
    Compute the masked Q-resolution in (Q, lambda) space.

    CleanSummedQ has been summed over pixels but not over wavelengths. This is exactly
    what is required for performing the remaining scaling and addition of the moderator
    term to obtain the Q-resolution. The result is still in (Q, lambda) space.
    """
    if mask is not None:
        pixel_term = mask_range(pixel_term, mask=mask)
    lambda2 = pixel_term.coords['wavelength'] ** 2
    Q2 = pixel_term.coords['Q'] ** 2
    resolution = (
        pi**2 / (3 * lambda2**2) * pixel_term + Q2 * moderator_time_spread**2 / lambda2
    )
    return QResolutionByWavelength(resolution)


def reduce_resolution_q(
    data: QResolutionByWavelength, bands: ProcessedWavelengthBands
) -> QResolution:
    # TODO Add op argument to allow injection of different reduction functions
    return QResolution(_reduce(data, bands, op=sc.max))


providers = (
    pixel_term,
    groupby_q_max,
    mask_and_compute_resolution_q,
    reduce_resolution_q,
)
