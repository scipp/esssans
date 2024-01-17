# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from typing import Optional

import scipp as sc
from scipp.core import concepts

from .types import (
    CalibratedMaskedData,
    CleanDirectBeam,
    CleanMonitor,
    CleanSummedQ,
    CleanWavelength,
    Denominator,
    DetectorPixelShape,
    EmptyBeamRun,
    Incident,
    IofQ,
    LabFrameTransform,
    NormWavelengthTerm,
    Numerator,
    RunType,
    SolidAngle,
    Transmission,
    TransmissionFraction,
    TransmissionRun,
    UncertaintyBroadcastMode,
)
from .uncertainty import (
    broadcast_with_upper_bound_variances,
    drop_variances_if_broadcast,
)


def solid_angle(
    data: CalibratedMaskedData[RunType],
    pixel_shape: DetectorPixelShape[RunType],
    transform: LabFrameTransform[RunType],
) -> SolidAngle[RunType]:
    """
    Solid angle for cylindrical pixels.

    Note that the approximation is valid when the distance from sample
    to pixel is much larger than the length or the radius of the pixels.

    Parameters
    ----------
    data:
        The DataArray that contains the positions of the detector pixels
        and the position of the sample in the coords.

    pixel_shape:
        Contains the description of the detector pixel shape.

    transform:
        Transformation from the local coordinate system of the detector
        to the coordinate system of the sample.

    Returns
    -------
    :
        The solid angle of the detector pixels, as viewed from the sample position.
    """
    face_1_center, face_1_edge, face_2_center = (
        pixel_shape['vertices']['vertex', i] for i in range(3)
    )
    cylinder_axis = transform * face_2_center - transform * face_1_center
    radius = sc.norm(face_1_center - face_1_edge)
    length = sc.norm(cylinder_axis)

    omega = _approximate_solid_angle_for_cylinder_shaped_pixel_of_detector(
        pixel_position=data.coords['position'] - data.coords['sample_position'],
        cylinder_axis=cylinder_axis,
        radius=radius,
        length=length,
    )
    return SolidAngle[RunType](
        concepts.rewrap_reduced_data(
            prototype=data, data=omega, dim=set(data.dims) - set(omega.dims)
        )
    )


def _approximate_solid_angle_for_cylinder_shaped_pixel_of_detector(
    pixel_position: sc.Variable,
    cylinder_axis: sc.Variable,
    radius: sc.Variable,
    length: sc.Variable,
):
    r"""Computes solid angle of a detector pixel under the assumption that the
    distance to the detector pixel from the sample is large compared to the radius
    and to the length of the piece of the detector cylinder that makes up the pixel.

    The cylinder is approximated by a rectangle element contained in the cylinder.
    The normal of the rectangle is selected to
    1. be orthogonal to the cylinder axis and
    2. "maximally parallel" to the pixel position vector.
    This occurs when the normal :math:`n`, the cylinder axis :math:`c` and the
    pixel position :math:`r` fall in a shared plane such that
    .. math::
        r = (n\cdot r)n + (c \cdot r)c.

    From the above relationship we get the scalar product of the position
    vector and the normal:

    .. math::
        |\hat{r}\cdot n| = \sqrt{\frac{r \cdot r - (c \cdot r^2)}{||r||^2}}

    The solid angle contribution from the detector element is approximated as
    .. math::
        \Delta\Omega =  A \frac{|\hat{r}\cdot n|}{||r||^2}
    where :math:`A = 2 R L` is the area of the rectangular element
    and :math:`\hat{r}` is the normalized pixel position vector.
    """
    norm_pp = sc.norm(pixel_position)
    norm_ca = sc.norm(cylinder_axis)
    cosalpha = sc.sqrt(
        1 - (sc.dot(pixel_position, cylinder_axis) / (norm_pp * norm_ca)) ** 2
    )
    return (2 * radius * length) * cosalpha / norm_pp**2


def transmission_fraction(
    sample_incident_monitor: CleanMonitor[TransmissionRun[RunType], Incident],
    sample_transmission_monitor: CleanMonitor[TransmissionRun[RunType], Transmission],
    direct_incident_monitor: CleanMonitor[EmptyBeamRun, Incident],
    direct_transmission_monitor: CleanMonitor[EmptyBeamRun, Transmission],
) -> TransmissionFraction[RunType]:
    """
    Approximation based on equations in
    `CalculateTransmission <https://docs.mantidproject.org/v4.0.0/algorithms/CalculateTransmission-v1.html>`_
    documentation:
    ``(sample_transmission_monitor / direct_transmission_monitor) * (direct_incident_monitor / sample_incident_monitor)``

    This is equivalent to ``mantid.CalculateTransmission`` without fitting.
    Inputs should be wavelength-dependent.

    Parameters
    ----------
    sample_incident_monitor:
        The incident monitor data for the sample (transmission) run.
    sample_transmission_monitor:
        The transmission monitor data for the sample (transmission) run.
    direct_incident_monitor:
        The incident monitor data for the direct beam run.
    direct_transmission_monitor:
        The transmission monitor data for the direct beam run.

    Returns
    -------
    :
        The transmission fraction computed from the monitor counts.
    """  # noqa: E501
    frac = (sample_transmission_monitor / direct_transmission_monitor) * (
        direct_incident_monitor / sample_incident_monitor
    )
    return TransmissionFraction[RunType](frac)


_broadcasters = {
    UncertaintyBroadcastMode.drop: drop_variances_if_broadcast,
    UncertaintyBroadcastMode.upper_bound: broadcast_with_upper_bound_variances,
    UncertaintyBroadcastMode.fail: lambda x, sizes: x,
}


def iofq_norm_wavelength_term(
    incident_monitor: CleanMonitor[RunType, Incident],
    transmission_fraction: TransmissionFraction[RunType],
    direct_beam: Optional[CleanDirectBeam],
    uncertainties: UncertaintyBroadcastMode,
) -> NormWavelengthTerm[RunType]:
    """
    Compute the wavelength-dependent contribution to the denominator term for the I(Q)
    normalization.
    Keeping this as a separate function allows us to compute it once during the
    iterations for finding the beam center, while the solid angle is recomputed
    for each iteration.

    This is basically:
    ``incident_monitor * transmission_fraction * direct_beam``
    If the direct beam is not supplied, it is assumed to be 1.

    Because the multiplication between the ``incident_monitor * transmission_fraction``
    (pixel-independent) and the direct beam (potentially pixel-dependent) consists of a
    broadcast operation which would introduce correlations, variances of the direct
    beam are dropped or replaced by an upper-bound estimation, depending on the
    configured mode.

    Parameters
    ----------
    incident_monitor:
        The incident monitor data (depends on wavelength).
    transmission_fraction:
        The transmission fraction (depends on wavelength).
    direct_beam:
        The direct beam function (depends on wavelength).
    uncertainties:
        The mode for broadcasting uncertainties. See
        :py:class:`UncertaintyBroadcastMode` for details.

    Returns
    -------
    :
        Wavelength-dependent term
        (incident_monitor * transmission_fraction * direct_beam) to be used for
        the denominator of the SANS I(Q) normalization.
        Used by :py:func:`iofq_denominator`.
    """
    out = incident_monitor * transmission_fraction
    if direct_beam is not None:
        broadcast = _broadcasters[uncertainties]
        out = direct_beam * broadcast(out, sizes=direct_beam.sizes)
    # Convert wavelength coordinate to midpoints for future histogramming
    out.coords['wavelength'] = sc.midpoints(out.coords['wavelength'])
    return NormWavelengthTerm[RunType](out)


def iofq_denominator(
    wavelength_term: NormWavelengthTerm[RunType],
    solid_angle: SolidAngle[RunType],
    uncertainties: UncertaintyBroadcastMode,
) -> CleanWavelength[RunType, Denominator]:
    """
    Compute the denominator term for the I(Q) normalization.

    In a SANS experiment, the scattering cross section :math:`I(Q)` is defined as
    (`Heenan et al. 1997 <https://doi.org/10.1107/S0021889897002173>`_):

    .. math::

       I(Q) = \\frac{\\partial\\Sigma{Q}}{\\partial\\Omega} = \\frac{A_{H} \\Sigma_{R,\\lambda\\subset Q} C(R, \\lambda)}{A_{M} t \\Sigma_{R,\\lambda\\subset Q}M(\\lambda)T(\\lambda)D(\\lambda)\\Omega(R)}

    where :math:`A_{H}` is the area of a mask (which avoids saturating the detector)
    placed between the monitor of area :math:`A_{M}` and the main detector.
    :math:`\\Omega` is the detector solid angle, and :math:`C` is the count rate on the
    main detector, which depends on the position :math:`R` and the wavelength.
    :math:`t` is the sample thickness, :math:`M` represents the incident monitor count
    rate for the sample run, and :math:`T` is known as the transmission fraction.

    Note that the incident monitor used to compute the transmission fraction is not
    necessarily the same as :math:`M`, as the transmission fraction is usually computed
    from a separate 'transmission' run (in the 'sample' run, the transmission monitor is
    commonly moved out of the beam path, to avoid polluting the sample detector signal).

    Finally, :math:`D` is the 'direct beam function', and is defined as

    .. math::

       D(\\lambda) = \\frac{\\eta(\\lambda)}{\\eta_{M}(\\lambda)} \\frac{A_{H}}{A_{M}}

    where :math:`\\eta` and :math:`\\eta_{M}` are the detector and monitor
    efficiencies, respectively.

    Hence, in order to normalize the main detector counts :math:`C`, we need compute the
    transmission fraction :math:`T(\\lambda)`, the direct beam function
    :math:`D(\\lambda)` and the solid angle :math:`\\Omega(R)`.

    The denominator is then simply:
    :math:`M_{\\lambda} T_{\\lambda} D_{\\lambda} \\Omega_{R}`,
    which is equivalent to ``wavelength_term * solid_angle``.
    The ``wavelength_term`` includes all but the ``solid_angle`` and is computed by
    :py:func:`iofq_norm_wavelength_term_sample` or
    :py:func:`iofq_norm_wavelength_term_background`.

    Because the multiplication between the wavelength dependent terms
    and the pixel dependent term (solid angle) consists of a broadcast operation which
    would introduce correlations, variances are dropped or replaced by an upper-bound
    estimation, depending on the configured mode.

    Parameters
    ----------
    wavelength_term:
        The term that depends on wavelength, computed by
        :py:func:`iofq_norm_wavelength_term`.
    solid_angle:
        The solid angle of the detector pixels, as viewed from the sample position.
    uncertainties:
        The mode for broadcasting uncertainties. See
        :py:class:`UncertaintyBroadcastMode` for details.

    Returns
    -------
    :
        The denominator for the SANS I(Q) normalization.
    """  # noqa: E501
    broadcast = _broadcasters[uncertainties]
    denominator = solid_angle * broadcast(wavelength_term, sizes=solid_angle.sizes)
    return CleanWavelength[RunType, Denominator](denominator)


def normalize(
    numerator: CleanSummedQ[RunType, Numerator],
    denominator: CleanSummedQ[RunType, Denominator],
) -> IofQ[RunType]:
    """
    Perform normalization of counts as a function of Q.
    If the numerator contains events, we use the sc.lookup function to perform the
    division.

    Parameters
    ----------
    numerator:
        The data whose counts will be divided by the denominator. This can either be
        event or dense (histogrammed) data.
    denominator:
        The divisor for the normalization operation. This cannot be event data, it must
        contain histogrammed data.

    Returns
    -------
    :
        The input data normalized by the supplied denominator.
    """
    if denominator.variances is not None and numerator.bins is not None:
        # Event-mode normalization is not correct of norm-term has variances.
        # See https://doi.org/10.3233/JNR-220049 for context.
        numerator = numerator.hist()
    if numerator.bins is not None:
        da = numerator.bins / sc.lookup(func=denominator, dim='Q')
    else:
        da = numerator / denominator
    return IofQ[RunType](da)


providers = (
    transmission_fraction,
    iofq_norm_wavelength_term,
    iofq_denominator,
    normalize,
    solid_angle,
)
