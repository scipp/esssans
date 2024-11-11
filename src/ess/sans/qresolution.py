# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""Q-resolution calculation for SANS data."""

from typing import NewType

import numpy as np
import scipp as sc
from scipp.constants import pi

from .common import mask_range
from .conversions import ElasticCoordTransformGraph
from .i_of_q import resample_direct_beam
from .normalization import _reduce
from .types import (
    CalibratedBeamline,
    DetectorMasks,
    DimsToKeep,
    ProcessedWavelengthBands,
    QBins,
    SampleRun,
    WavelengthBins,
    WavelengthMask,
)

DeltaR = NewType("DeltaR", sc.Variable)
"""Virtual ring width on the detector."""

SampleApertureRadius = NewType("SampleApertureRadius", sc.Variable)
"""Sample aperture radius, R2."""
SourceApertureRadius = NewType("SourceApertureRadius", sc.Variable)
"""Source aperture radius, R1."""
SigmaModerator = NewType("SigmaModerator", sc.DataArray)
"""
Moderator wavelength spread as a function of wavelength.

This is derived from ModeratorTimeSpread.
"""
CollimationLength = NewType("CollimationLength", sc.Variable)
"""Collimation length."""
ModeratorTimeSpreadFilename = NewType("ModeratorTimeSpreadFilename", str)
ModeratorTimeSpread = NewType("ModeratorTimeSpread", sc.DataArray)
"""Moderator time-spread as a function of wavelength."""


QResolutionByPixel = NewType("QResolutionByPixel", sc.DataArray)
QResolutionByWavelength = NewType("QResolutionByWavelength", sc.DataArray)
QResolution = NewType("QResolution", sc.DataArray)


def load_isis_moderator_time_spread(
    filename: ModeratorTimeSpreadFilename,
) -> ModeratorTimeSpread:
    """
    Load moderator time spread from an ISIS moderator file.

    Files looks as follows:

    .. code-block:: text

         Fri 08-Aug-2015, LET exptl data (FWHM/2.35) [...]

          61    0    0    0    1   61    0
                0         0         0         0
        3 (F12.5,2E14.6)
            0.00000  2.257600E+01  0.000000E+00
            0.50000  2.677152E+01  0.000000E+00
            1.00000  3.093920E+01  0.000000E+00
            1.50000  3.507903E+01  0.000000E+00
            2.00000  3.919100E+01  0.000000E+00

    The first column is the wavelength in Angstrom, the second is the time spread in
    microseconds. The third column is the error on the time spread, which we ignore.
    """
    wavelength, time_spread = np.loadtxt(
        filename, skiprows=5, usecols=(0, 1), unpack=True
    )
    wav = 'wavelength'
    return ModeratorTimeSpread(
        sc.DataArray(
            data=sc.array(dims=[wav], values=time_spread, unit='us'),
            coords={wav: sc.array(dims=[wav], values=wavelength, unit='angstrom')},
        )
    )


def moderator_time_spread_to_wavelength_spread(
    moderator_time_spread: ModeratorTimeSpread,
    beamline: CalibratedBeamline[SampleRun],
    graph: ElasticCoordTransformGraph,
    wavelength_bins: WavelengthBins,
    masks: DetectorMasks,
) -> SigmaModerator:
    """
    Convert the moderator time spread to a wavelength spread and mask pixels.

    This resamples the raw moderator time spread to the wavelength bins used in the data
    reduction. The result is a DataArray with the time spread as a function of pixel and
    wavelength.

    Parameters
    ----------
    moderator_time_spread:
        Moderator time spread.
    beamline:
        Beamline geometry information required for converting time spread to wavelength
        spread. The latter depends on the pixel position via sample-detector distance
        L2.
    graph:
        Coordinate transformation graph for elastic scattering for computing wavelength.
    wavelength_bins:
        Binning in wavelength.
    masks:
        Detector masks.

    Returns
    -------
    :
        Wavelength spread of the moderator.
    """
    dtof = resample_direct_beam(moderator_time_spread, wavelength_bins=wavelength_bins)
    # We would like to "transform" the *data*, but we only have transform_coords, so
    # there is some back and forth between data and coords here.
    dummy = beamline.broadcast(sizes={**beamline.sizes, **dtof.sizes})
    dummy.data = sc.empty(sizes=dummy.sizes)
    da = (
        dummy.assign_coords(tof=dtof.data)
        .assign_masks(masks)
        .transform_coords('wavelength', graph=graph, keep_inputs=False)
    )
    da.data = da.coords.pop('wavelength')
    return SigmaModerator(da.assign_coords(wavelength=wavelength_bins))


def q_resolution_by_pixel(
    sigma_moderator: SigmaModerator,
    source_aperture: SourceApertureRadius,
    sample_aperture: SampleApertureRadius,
    collimation_length: CollimationLength,
    delta_r: DeltaR,
    graph: ElasticCoordTransformGraph,
) -> QResolutionByPixel:
    """
    Calculate the Q-resolution per pixel and wavelength.

    Parameters
    ----------
    sigma_moderator:
        Moderator wavelength spread as a function of pixel and wavelength.
    source_aperture:
        Source aperture radius, R1.
    sample_aperture:
        Sample aperture radius, R2.
    collimation_length:
        Collimation length.
    delta_r:
        Virtual ring width on the detector.
    graph:
        Coordinate transformation graph for elastic scattering.

    Returns
    -------
    :
        Q-resolution per pixel and wavelength.
    """
    wavelength_bins = sigma_moderator.coords['wavelength']
    delta_lambda = wavelength_bins[1:] - wavelength_bins[:-1]
    result = sigma_moderator.assign_coords(
        wavelength=sc.midpoints(wavelength_bins)
    ).transform_coords('Q', graph=graph, keep_inputs=True)
    L2 = result.coords["L2"]
    L3 = sc.reciprocal(sc.reciprocal(collimation_length) + sc.reciprocal(L2))
    pixel_term = (
        3 * ((source_aperture / collimation_length) ** 2 + (sample_aperture / L3) ** 2)
        + (delta_r / L2) ** 2
    )
    sigma_lambda2 = result.data**2 + delta_lambda**2 / 12
    result.data = (pi**2 / 3) * pixel_term + result.coords['Q'] ** 2 * sigma_lambda2
    inv_lambda = sc.reciprocal(result.coords['wavelength'])
    return QResolutionByPixel(sc.sqrt(result) * inv_lambda)


def q_resolution_by_wavelength(
    data: QResolutionByPixel,
    q_bins: QBins,
    dims_to_keep: DimsToKeep,
    mask: WavelengthMask,
) -> QResolutionByWavelength:
    """
    Compute the masked Q-resolution in (Q, lambda) space.

    Parameters
    ----------
    data:
        Q-resolution per pixel and wavelength.
    q_bins:
        Binning in Q.
    dims_to_keep:
        Detector dimensions that should not be reduced.
    mask:
        Wavelength mask.

    Returns
    -------
    :
        Q-resolution in (Q, lambda) space.
    """
    dims = [dim for dim in data.dims if dim not in (*dims_to_keep, 'wavelength')]
    resolution = data.bin(Q=q_bins, dim=dims)
    if mask is not None:
        resolution = mask_range(resolution, mask=mask)
    return QResolutionByWavelength(resolution)


def reduce_resolution_q(
    data: QResolutionByWavelength, bands: ProcessedWavelengthBands
) -> QResolution:
    """
    Q-resolution reduced over wavelength dimension.

    The result depends only Q, or (Q, wavelength_band) if wavelength bands are used.

    Note that the result is *binned data*, with each bin entry representing a specific
    input pixel and wavelength. The final solution can be obtained, e.g., by computing
    `resolution.bins.max()`, assuming `resolution is the output of this function.

    Parameters
    ----------
    data:
        Q-resolution in (Q, lambda) space.
    bands:
        Wavelength bands.

    Returns
    -------
    :
        Reduced Q-resolution.
    """
    return QResolution(_reduce(data, bands=bands))


providers = (
    load_isis_moderator_time_spread,
    moderator_time_spread_to_wavelength_spread,
    q_resolution_by_pixel,
    q_resolution_by_wavelength,
    reduce_resolution_q,
)
