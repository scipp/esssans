# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc
from ess.reduce.uncertainty import UncertaintyBroadcastMode, broadcast_uncertainties
from scipp.core import concepts

from .types import (
    CalibratedDetector,
    CleanDirectBeam,
    CleanMonitor,
    CleanSummedQ,
    CleanSummedQxy,
    CleanWavelength,
    Denominator,
    DetectorMasks,
    DetectorPixelShape,
    EmptyBeamRun,
    Incident,
    IofQ,
    IofQPart,
    IofQxy,
    LabFrameTransform,
    MaskedSolidAngle,
    NormWavelengthTerm,
    Numerator,
    ProcessedWavelengthBands,
    ReducedQ,
    ReducedQxy,
    ReturnEvents,
    ScatteringRunType,
    SolidAngle,
    Transmission,
    TransmissionFraction,
    TransmissionRun,
    WavelengthBands,
    WavelengthBins,
)


def solid_angle(
    data: CalibratedDetector[ScatteringRunType],
    pixel_shape: DetectorPixelShape[ScatteringRunType],
    transform: LabFrameTransform[ScatteringRunType],
) -> SolidAngle[ScatteringRunType]:
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
    return SolidAngle[ScatteringRunType](
        concepts.rewrap_reduced_data(
            prototype=data, data=omega, dim=set(data.dims) - set(omega.dims)
        )
    )


def mask_solid_angle(
    solid_angle: SolidAngle[ScatteringRunType],
    masks: DetectorMasks,
) -> MaskedSolidAngle[ScatteringRunType]:
    return MaskedSolidAngle[ScatteringRunType](solid_angle.assign_masks(masks))


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
    sample_incident_monitor: CleanMonitor[TransmissionRun[ScatteringRunType], Incident],
    sample_transmission_monitor: CleanMonitor[
        TransmissionRun[ScatteringRunType], Transmission
    ],
    direct_incident_monitor: CleanMonitor[EmptyBeamRun, Incident],
    direct_transmission_monitor: CleanMonitor[EmptyBeamRun, Transmission],
) -> TransmissionFraction[ScatteringRunType]:
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
    return TransmissionFraction[ScatteringRunType](frac)


def iofq_norm_wavelength_term(
    incident_monitor: CleanMonitor[ScatteringRunType, Incident],
    transmission_fraction: TransmissionFraction[ScatteringRunType],
    direct_beam: CleanDirectBeam,
    uncertainties: UncertaintyBroadcastMode,
) -> NormWavelengthTerm[ScatteringRunType]:
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
        :py:class:`ess.reduce.uncertainty.UncertaintyBroadcastMode` for details.

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
        # Make wavelength the inner dim
        dims = list(direct_beam.dims)
        dims.remove('wavelength')
        dims.append('wavelength')
        direct_beam = direct_beam.transpose(dims)
        out = direct_beam * broadcast_uncertainties(
            out, prototype=direct_beam, mode=uncertainties
        )
    # Convert wavelength coordinate to midpoints for future histogramming
    out.coords['wavelength'] = sc.midpoints(out.coords['wavelength'])
    return NormWavelengthTerm[ScatteringRunType](out)


def iofq_denominator(
    wavelength_term: NormWavelengthTerm[ScatteringRunType],
    solid_angle: MaskedSolidAngle[ScatteringRunType],
    uncertainties: UncertaintyBroadcastMode,
) -> CleanWavelength[ScatteringRunType, Denominator]:
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
        :py:class:`ess.reduce.uncertainty.UncertaintyBroadcastMode` for details.

    Returns
    -------
    :
        The denominator for the SANS I(Q) normalization.
    """  # noqa: E501
    denominator = solid_angle * broadcast_uncertainties(
        wavelength_term, prototype=solid_angle, mode=uncertainties
    )
    return CleanWavelength[ScatteringRunType, Denominator](denominator)


def process_wavelength_bands(
    wavelength_bands: WavelengthBands,
    wavelength_bins: WavelengthBins,
) -> ProcessedWavelengthBands:
    """
    Perform some checks and potential reshaping on the wavelength bands.

    The wavelength bands must be either one- or two-dimensional.
    If the wavelength bands are defined as a one-dimensional array, convert them to a
    two-dimensional array with start and end wavelengths.

    The final bands must have a size of 2 in the wavelength dimension, defining a start
    and an end wavelength.
    """
    if wavelength_bands is None:
        wavelength_bands = sc.concat(
            [wavelength_bins.min(), wavelength_bins.max()], dim='wavelength'
        )
    if wavelength_bands.ndim == 1:
        wavelength_bands = sc.concat(
            [wavelength_bands[:-1], wavelength_bands[1:]], dim='x'
        ).rename(x='wavelength', wavelength='band')
    if wavelength_bands.ndim != 2:
        raise ValueError(
            'Wavelength_bands must be one- or two-dimensional, '
            f'got {wavelength_bands.ndim}.'
        )
    if wavelength_bands.sizes['wavelength'] != 2:
        raise ValueError(
            'Wavelength_bands must have a size of 2 in the wavelength dimension, '
            'defining a start and an end wavelength, '
            f'got {wavelength_bands.sizes["wavelength"]}.'
        )
    return wavelength_bands


def _normalize(
    numerator: sc.DataArray,
    denominator: sc.DataArray,
    return_events: ReturnEvents,
    uncertainties: UncertaintyBroadcastMode,
) -> sc.DataArray:
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
    return_events:
        Whether to return the result as event data or histogrammed data.
    uncertainties:
        The mode for broadcasting uncertainties. See
        :py:class:`ess.reduce.uncertainty.UncertaintyBroadcastMode` for details.

    Returns
    -------
    :
        The input data normalized by the supplied denominator.
    """
    if return_events and numerator.bins is not None:
        # Naive event-mode normalization is not correct if norm-term has variances.
        # See https://doi.org/10.3233/JNR-220049 for context.
        denominator = broadcast_uncertainties(
            denominator, prototype=numerator, mode=uncertainties
        )
    elif numerator.bins is not None:
        numerator = numerator.hist()
    numerator /= denominator.drop_coords(
        [name for name in denominator.coords if name not in denominator.dims]
    )
    return numerator


def _do_reduce(da: sc.DataArray) -> sc.DataArray:
    wav = 'wavelength'
    if da.sizes[wav] == 1:  # Can avoid costly event-data da.bins.concat
        return da.squeeze(wav)
    return da.sum(wav) if da.bins is None else da.bins.concat(wav)


def _reduce(part: sc.DataArray, /, *, bands: ProcessedWavelengthBands) -> sc.DataArray:
    """
    Reduce data by summing or concatenating along the wavelength dimension.

    Parameters
    ----------
    data:
        Numerator or denominator data to be reduced.
    wavelength_bands:
        Defines bands in wavelength that can be used to separate different wavelength
        ranges that contribute to different regions in Q space. Note that this needs to
        be defined, so if all wavelengths should be used, this should simply be a start
        and end edges that encompass the entire wavelength range.

    Returns
    -------
    :
        Q-dependent data, ready for normalization.
    """
    wav = 'wavelength'
    if part.bins is not None:
        # If in event mode the desired wavelength binning has not been applied, we need
        # it for splitting by bands, or restricting the range in case of a single band.
        part = part.bin(wavelength=sc.sort(bands.flatten(to=wav), wav))
    parts = [
        _do_reduce(part[wav, wav_range[0] : wav_range[1]])
        for wav_range in sc.collapse(bands, keep=wav).values()
    ]
    band_dim = (set(bands.dims) - {'wavelength'}).pop()
    reduced = parts[0] if len(parts) == 1 else sc.concat(parts, band_dim)
    return reduced.assign_coords(wavelength=bands.squeeze())


def reduce_q(
    data: CleanSummedQ[ScatteringRunType, IofQPart], bands: ProcessedWavelengthBands
) -> ReducedQ[ScatteringRunType, IofQPart]:
    return ReducedQ[ScatteringRunType, IofQPart](_reduce(data, bands=bands))


def reduce_qxy(
    data: CleanSummedQxy[ScatteringRunType, IofQPart], bands: ProcessedWavelengthBands
) -> ReducedQxy[ScatteringRunType, IofQPart]:
    return ReducedQxy[ScatteringRunType, IofQPart](_reduce(data, bands=bands))


def normalize_q(
    numerator: ReducedQ[ScatteringRunType, Numerator],
    denominator: ReducedQ[ScatteringRunType, Denominator],
    return_events: ReturnEvents,
    uncertainties: UncertaintyBroadcastMode,
) -> IofQ[ScatteringRunType]:
    return IofQ[ScatteringRunType](
        _normalize(
            numerator=numerator,
            denominator=denominator,
            return_events=return_events,
            uncertainties=uncertainties,
        )
    )


def normalize_qxy(
    numerator: ReducedQxy[ScatteringRunType, Numerator],
    denominator: ReducedQxy[ScatteringRunType, Denominator],
    return_events: ReturnEvents,
    uncertainties: UncertaintyBroadcastMode,
) -> IofQxy[ScatteringRunType]:
    return IofQxy[ScatteringRunType](
        _normalize(
            numerator=numerator,
            denominator=denominator,
            return_events=return_events,
            uncertainties=uncertainties,
        )
    )


reduce_q.__doc__ = _reduce.__doc__
reduce_qxy.__doc__ = _reduce.__doc__
normalize_q.__doc__ = _normalize.__doc__
normalize_qxy.__doc__ = _normalize.__doc__


providers = (
    transmission_fraction,
    iofq_norm_wavelength_term,
    iofq_denominator,
    reduce_q,
    reduce_qxy,
    normalize_q,
    normalize_qxy,
    process_wavelength_bands,
    solid_angle,
    mask_solid_angle,
)
