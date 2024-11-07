# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""Q-resolution calculation for SANS data."""

from typing import NewType

import scipp as sc

from .types import CleanQ, Denominator, Resolution, SampleRun

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

QResolutionDetectorTerm = NewType("QResolutionDetectorTerm", sc.DataArray)


def pixel_term(
    detector: CleanQ[SampleRun, Denominator],
    delta_r: DeltaR,
    sample_aperture: SampleApertureRadius,
    source_aperture: SourceApertureRadius,
    collimation_length: CollimationLength,
) -> CleanQ[SampleRun, Resolution]:
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
    #
    result = detector.copy(deep=False)
    result.data = (
        3 * ((source_aperture / collimation_length) ** 2 + (sample_aperture / L3) ** 2)
        + (delta_r / L2) ** 2
    )
    return CleanQ[SampleRun, Resolution](result)
