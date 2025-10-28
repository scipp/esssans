# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import NewType

import scipp as sc
from scippneutron.conversion.beamline import (
    beam_aligned_unit_vectors,
    scattering_angles_with_gravity,
)
from scippneutron.conversion.graph import beamline, tof

from ess.reduce.uncertainty import broadcast_uncertainties

from .common import mask_range
from .types import (
    CorrectedQ,
    CorrectedQxy,
    CorrectedQ,
    CorrectedQxy,
    CorrectedDetector,
    CorrectForGravity,
    Denominator,
    DimsToKeep,
    IntensityQPart,
    MaskedData,
    MonitorTerm,
    MonitorType,
    Numerator,
    QBins,
    QxBins,
    QyBins,
    RunType,
    ScatteringRunType,
    TofMonitor,
    UncertaintyBroadcastMode,
    WavelengthMask,
    WavelengthMonitor,
    WavelengthScaledQ,
    WavelengthScaledQxy,
)


def cyl_unit_vectors(incident_beam: sc.Variable, gravity: sc.Variable):
    vectors = beam_aligned_unit_vectors(incident_beam=incident_beam, gravity=gravity)
    return {
        'cyl_x_unit_vector': vectors['beam_aligned_unit_x'],
        'cyl_y_unit_vector': vectors['beam_aligned_unit_y'],
    }


def cylindrical_x(
    cyl_x_unit_vector: sc.Variable, scattered_beam: sc.Variable
) -> sc.Variable:
    """
    Compute the horizontal x coordinate perpendicular to the incident beam direction.
    Note that it is assumed here that the incident beam is perpendicular to the gravity
    vector.
    """
    return sc.dot(scattered_beam, cyl_x_unit_vector)


def cylindrical_y(
    cyl_y_unit_vector: sc.Variable, scattered_beam: sc.Variable
) -> sc.Variable:
    """
    Compute the vertical y coordinate perpendicular to the incident beam direction.
    Note that it is assumed here that the incident beam is perpendicular to the gravity
    vector.
    """
    return sc.dot(scattered_beam, cyl_y_unit_vector)


def phi_no_gravity(
    cylindrical_x: sc.Variable, cylindrical_y: sc.Variable
) -> sc.Variable:
    """
    Compute the cylindrical phi angle around the incident beam. Note that it is assumed
    here that the incident beam is perpendicular to the gravity vector.
    """
    return sc.atan2(y=cylindrical_y, x=cylindrical_x)


def Qxy(Q: sc.Variable, phi: sc.Variable) -> dict[str, sc.Variable]:
    """
    Compute the Qx and Qy components of the scattering vector from the scattering angle,
    wavelength, and phi angle.
    """
    Qx = sc.cos(phi)
    Qy = sc.sin(phi)
    if Q.bins is not None and phi.bins is not None:
        Qx *= Q
        Qy *= Q
    else:
        Qx = Qx * Q
        Qy = Qy * Q
    return {'Qx': Qx, 'Qy': Qy}


ElasticCoordTransformGraph = NewType('ElasticCoordTransformGraph', dict)
MonitorCoordTransformGraph = NewType('MonitorCoordTransformGraph', dict)


def sans_elastic(correct_for_gravity: CorrectForGravity) -> ElasticCoordTransformGraph:
    """
    Generate a coordinate transformation graph for SANS elastic scattering.

    It is based on classical conversions from ``tof`` and pixel ``position`` to
    :math:`\\lambda` (``wavelength``), :math:`\\theta` (``theta``) and
    :math:`Q` (``Q``), but can take into account the Earth's gravitational field,
    which bends the flight path of the neutrons, to compute the scattering angle
    :math:`\\theta`.

    The angle can be found using the following expression
    (`Seeger & Hjelm 1991 <https://doi.org/10.1107/S0021889891004764>`_):

    .. math::

       \\theta = \\frac{1}{2}\\sin^{-1}\\left(\\frac{\\sqrt{ x^{2} + \\left( y + \\frac{g m_{\\rm n}}{2 h^{2}} \\lambda^{2} L_{2}^{2} \\right)^{2} } }{L_{2}}\\right)

    where :math:`x` and :math:`y` are the spatial coordinates of the pixels in the
    horizontal and vertical directions, respectively,
    :math:`m_{\\rm n}` is the neutron mass,
    :math:`L_{2}` is the distance between the sample and a detector pixel,
    :math:`g` is the acceleration due to gravity,
    and :math:`h` is Planck's constant.

    By default, the effects of gravity on the neutron flight paths are not included
    (equivalent to :math:`g = 0` in the expression above).

    Parameters
    ----------
    correct_for_gravity:
        Take into account the bending of the neutron flight paths from the
        Earth's gravitational field if ``True``.
    """  # noqa: E501
    graph = {**beamline.beamline(scatter=True), **tof.elastic_Q('tof')}
    if correct_for_gravity:
        del graph['two_theta']
        graph[('two_theta', 'phi')] = scattering_angles_with_gravity
    else:
        graph['phi'] = phi_no_gravity
    graph[('cyl_x_unit_vector', 'cyl_y_unit_vector')] = cyl_unit_vectors
    graph['cylindrical_x'] = cylindrical_x
    graph['cylindrical_y'] = cylindrical_y
    graph[('Qx', 'Qy')] = Qxy
    return ElasticCoordTransformGraph(graph)


def sans_monitor() -> MonitorCoordTransformGraph:
    """
    Generate a coordinate transformation graph for SANS monitor (no scattering).
    """
    return MonitorCoordTransformGraph(
        {**beamline.beamline(scatter=False), **tof.elastic_wavelength('tof')}
    )


def monitor_to_wavelength(
    monitor: TofMonitor[RunType, MonitorType], graph: MonitorCoordTransformGraph
) -> WavelengthMonitor[RunType, MonitorType]:
    return WavelengthMonitor[RunType, MonitorType](
        monitor.transform_coords('wavelength', graph=graph, keep_inputs=False)
    )


# TODO This demonstrates a problem: Transforming to wavelength should be possible
# for RawData, MaskedData, ... no reason to restrict necessarily.
# Would we be fine with just choosing on option, or will this get in the way for users?
def detector_to_wavelength(
    detector: MaskedData[ScatteringRunType],
    graph: ElasticCoordTransformGraph,
) -> CorrectedDetector[ScatteringRunType, Numerator]:
    return CorrectedDetector[ScatteringRunType, Numerator](
        detector.transform_coords('wavelength', graph=graph, keep_inputs=False)
    )


def mask_wavelength_q(
    da: CorrectedQ[ScatteringRunType, Numerator], mask: WavelengthMask
) -> WavelengthScaledQ[ScatteringRunType, Numerator]:
    if mask is not None:
        da = mask_range(da, mask=mask)
    return WavelengthScaledQ[ScatteringRunType, Numerator](da)


def mask_wavelength_qxy(
    da: CorrectedQxy[ScatteringRunType, Numerator], mask: WavelengthMask
) -> WavelengthScaledQxy[ScatteringRunType, Numerator]:
    if mask is not None:
        da = mask_range(da, mask=mask)
    return WavelengthScaledQxy[ScatteringRunType, Numerator](da)


def mask_and_scale_wavelength_q(
    da: CorrectedQ[ScatteringRunType, Denominator],
    mask: WavelengthMask,
    wavelength_term: MonitorTerm[ScatteringRunType],
    uncertainties: UncertaintyBroadcastMode,
) -> WavelengthScaledQ[ScatteringRunType, Denominator]:
    da = da * broadcast_uncertainties(wavelength_term, prototype=da, mode=uncertainties)
    if mask is not None:
        da = mask_range(da, mask=mask)
    return WavelengthScaledQ[ScatteringRunType, Denominator](da)


def mask_and_scale_wavelength_qxy(
    da: CorrectedQxy[ScatteringRunType, Denominator],
    mask: WavelengthMask,
    wavelength_term: MonitorTerm[ScatteringRunType],
    uncertainties: UncertaintyBroadcastMode,
) -> WavelengthScaledQxy[ScatteringRunType, Denominator]:
    da = da * broadcast_uncertainties(wavelength_term, prototype=da, mode=uncertainties)
    if mask is not None:
        da = mask_range(da, mask=mask)
    return WavelengthScaledQxy[ScatteringRunType, Denominator](da)


def _compute_Q(
    data: sc.DataArray,
    graph: ElasticCoordTransformGraph,
    target: tuple[str, ...],
    edges: dict[str, sc.Variable],
    dims_to_keep: tuple[str, ...],
) -> sc.DataArray:
    # Keep naming of wavelength dim, subsequent steps use a (Q[xy], wavelength) binning.
    data_q = data.transform_coords(
        target,
        graph=graph,
        keep_intermediate=False,
        rename_dims=False,
    )
    dims_to_reduce = set(data_q.dims) - {'wavelength'} - set(dims_to_keep or ())
    return (data_q.hist if data_q.bins is None else data_q.bin)(
        **edges, dim=dims_to_reduce
    )


def compute_Q(
    data: CorrectedDetector[ScatteringRunType, IntensityQPart],
    q_bins: QBins,
    dims_to_keep: DimsToKeep,
    graph: ElasticCoordTransformGraph,
) -> CorrectedQ[ScatteringRunType, IntensityQPart]:
    """
    Convert a data array from wavelength to Q.
    We then combine data from all pixels into a single I(Q) spectrum:

    * In the case of event data, events in all bins are concatenated
    * In the case of dense data, counts in all spectra are summed

    Parameters
    ----------
    data:
        A DataArray containing the data that is to be converted to Q.
    q_bins:
        The binning in Q to be used.
    dims_to_keep:
        Dimensions that should not be reduced and thus still be present in the final
        I(Q) result (this is typically the layer dimension).
    graph:
        The coordinate transformation graph to use.

    Returns
    -------
    :
        The input data converted to Q and then summed over all detector pixels.
    """
    return CorrectedQ[ScatteringRunType, IntensityQPart](
        _compute_Q(
            data=data,
            graph=graph,
            target=('Q',),
            edges={'Q': q_bins},
            dims_to_keep=dims_to_keep,
        )
    )


def compute_Qxy(
    data: CorrectedDetector[ScatteringRunType, IntensityQPart],
    qx_bins: QxBins,
    qy_bins: QyBins,
    dims_to_keep: DimsToKeep,
    graph: ElasticCoordTransformGraph,
) -> CorrectedQxy[ScatteringRunType, IntensityQPart]:
    """
    Convert a data array from wavelength to Qx and Qy.
    We then combine data from all pixels into a single I(Qx, Qy) spectrum:

    * In the case of event data, events in all bins are concatenated
    * In the case of dense data, counts in all spectra are summed

    Parameters
    ----------
    data:
        A DataArray containing the data that is to be converted to Q.
    qx_bins:
        The binning in Qx to be used.
    qy_bins:
        The binning in Qy to be used.
    dims_to_keep:
        Dimensions that should not be reduced and thus still be present in the final
        I(Qx, Qy) result (this is typically the layer dimension).
    graph:
        The coordinate transformation graph to use.

    Returns
    -------
    :
        The input data converted to Qx and Qy and then summed over all detector pixels.
    """
    return CorrectedQxy[ScatteringRunType, IntensityQPart](
        _compute_Q(
            data=data,
            graph=graph,
            target=('Qx', 'Qy'),
            edges={'Qy': qy_bins, 'Qx': qx_bins},
            dims_to_keep=dims_to_keep,
        )
    )


providers = (
    sans_elastic,
    sans_monitor,
    monitor_to_wavelength,
    detector_to_wavelength,
    mask_wavelength_q,
    mask_wavelength_qxy,
    mask_and_scale_wavelength_q,
    mask_and_scale_wavelength_qxy,
    compute_Q,
    compute_Qxy,
)
