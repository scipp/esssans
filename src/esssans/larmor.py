# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Loading and masking specific to the ISIS Sans2d instrument and files stored in Scipp's
HDF5 format.
"""
from functools import lru_cache
from typing import NewType, Optional

import sciline
import scipp as sc
import scippneutron as scn

from .common import gravity_vector
from .types import (
    BackgroundRun,
    CalibratedMaskedData,
    CleanMasked,
    EmptyBeamRun,
    Filename,
    Incident,
    MaskedData,
    NeXusMonitorName,
    Numerator,
    RawData,
    RawMonitor,
    RunType,
    SampleRun,
    SampleTransmissionRun,
    Transmission,
)

DetectorStrawMask = NewType('DetectorStrawMask', sc.Variable)
"""Detector straw mask"""
DetectorBeamStopMask = NewType('DetectorBeamStopMask', sc.Variable)
"""Detector beam stop mask"""
DetectorTubeEdgeMask = NewType('DetectorTubeEdgeMask', sc.Variable)
"""Detector tube edge mask"""


class DataAsStraws(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Data reshaped to have a straw dimension"""


@lru_cache
def load_larmor_run(filename: Filename[RunType]) -> RawData[RunType]:
    da = scn.load_nexus(filename)
    if 'gravity' not in da.coords:
        da.coords["gravity"] = gravity_vector()
    if 'sample_position' not in da.coords:
        da.coords['sample_position'] = sc.vector([0, 0, 0], unit='m')
    da.bins.constituents['data'].variances = da.bins.constituents['data'].values
    for name in ('monitor_1', 'monitor_2'):
        monitor = da.attrs[name].value
        if 'source_position' not in monitor.coords:
            monitor.coords["source_position"] = da.coords['source_position']
        monitor.values[0].variances = monitor.values[0].values
    pixel_shape = da.coords['pixel_shape'].values[0]
    da.coords['pixel_width'] = sc.norm(
        pixel_shape['face1_edge'] - pixel_shape['face1_center']
    ).data
    da.coords['pixel_height'] = sc.norm(
        pixel_shape['face2_center'] - pixel_shape['face1_center']
    ).data
    return RawData[RunType](da)


def get_empty_beam_transmission_monitor(
    da: RawData[EmptyBeamRun], nexus_name: NeXusMonitorName[Transmission]
) -> RawMonitor[EmptyBeamRun, Transmission]:
    # See https://github.com/scipp/sciline/issues/52 why copy needed
    mon = da.attrs[nexus_name].value.copy()
    return RawMonitor[EmptyBeamRun, Transmission](mon)


def get_empty_beam_incident_monitor(
    da: RawData[EmptyBeamRun], nexus_name: NeXusMonitorName[Incident]
) -> RawMonitor[EmptyBeamRun, Incident]:
    # See https://github.com/scipp/sciline/issues/52 why copy needed
    mon = da.attrs[nexus_name].value.copy()
    return RawMonitor[EmptyBeamRun, Incident](mon)


def get_background_transmission_monitor(
    da: RawData[BackgroundRun], nexus_name: NeXusMonitorName[Transmission]
) -> RawMonitor[BackgroundRun, Transmission]:
    # See https://github.com/scipp/sciline/issues/52 why copy needed
    mon = da.attrs[nexus_name].value.copy()
    return RawMonitor[BackgroundRun, Transmission](mon)


def get_sample_incident_monitor(
    da: RawData[SampleRun], nexus_name: NeXusMonitorName[Incident]
) -> RawMonitor[SampleRun, Incident]:
    # See https://github.com/scipp/sciline/issues/52 why copy needed
    mon = da.attrs[nexus_name].value.copy()
    return RawMonitor[SampleRun, Incident](mon)


def get_sample_transmission_monitor(
    da: RawData[SampleTransmissionRun], nexus_name: NeXusMonitorName[Transmission]
) -> RawMonitor[SampleRun, Transmission]:
    # See https://github.com/scipp/sciline/issues/52 why copy needed
    mon = da.attrs[nexus_name].value.copy()
    return RawMonitor[SampleRun, Transmission](mon)


def to_straws(da: RawData[RunType]) -> DataAsStraws[RunType]:
    # return DataAsStraws[RunType](
    #     da.fold(
    #         dim='detector_id', sizes=dict(layer=4, tube=32, straw=7, pixel=512)
    #     ).flatten(dims=['layer', 'tube', 'straw'], to='straw')
    # )
    return DataAsStraws[RunType](
        da.fold(
            dim='detector_id', sizes=dict(layer=4, tube=32, straw=7, pixel=512)
        ).flatten(dims=['tube', 'straw'], to='straw')
    )


def detector_straw_mask(
    sample_straws: CalibratedMaskedData[SampleRun],
) -> DetectorStrawMask:
    return DetectorStrawMask(
        sample_straws.sum(['tof', 'pixel']).data < sc.scalar(300.0, unit='counts')
    )


def detector_beam_stop_mask(
    sample_straws: CalibratedMaskedData[SampleRun],
) -> DetectorBeamStopMask:
    pos = sample_straws.coords['position']
    x = pos.fields.x
    y = pos.fields.y
    return DetectorBeamStopMask(
        (abs(x) < sc.scalar(0.03, unit='m')) & (abs(y) < sc.scalar(0.025, unit='m'))
    )


def detector_tube_edge_mask(
    sample_straws: CalibratedMaskedData[SampleRun],
) -> DetectorTubeEdgeMask:
    return DetectorTubeEdgeMask(
        abs(sample_straws.coords['position'].fields.x) > sc.scalar(0.36, unit='m')
    )


def mask_detectors(
    da: DataAsStraws[RunType],
) -> MaskedData[RunType]:
    """Apply pixel-specific masks to raw data.

    Parameters
    ----------
    da:
        Raw data.
    edge_mask:
        Mask for detector edges.
    holder_mask:
        Mask for sample holder.
    """
    return MaskedData[RunType](da)


def mask_after_calibration(
    da: CalibratedMaskedData[RunType],
    straw_mask: Optional[DetectorStrawMask],
    beam_stop_mask: Optional[DetectorBeamStopMask],
    tube_edge_mask: Optional[DetectorTubeEdgeMask],
) -> CleanMasked[RunType, Numerator]:
    """Apply pixel-specific masks to raw data.

    Parameters
    ----------
    da:
        Raw data.
    edge_mask:
        Mask for detector edges.
    holder_mask:
        Mask for sample holder.
    """
    da = da.copy(deep=False)
    if straw_mask is not None:
        da.masks['bad_straws'] = straw_mask
    if beam_stop_mask is not None:
        da.masks['beam_stop'] = beam_stop_mask
    if tube_edge_mask is not None:
        da.masks['tube_edges'] = tube_edge_mask
    return CleanMasked[RunType, Numerator](da)


providers = [
    load_larmor_run,
    to_straws,
    detector_straw_mask,
    detector_beam_stop_mask,
    detector_tube_edge_mask,
    get_empty_beam_incident_monitor,
    get_empty_beam_transmission_monitor,
    get_background_transmission_monitor,
    get_sample_incident_monitor,
    get_sample_transmission_monitor,
    mask_detectors,
    mask_after_calibration,
]
"""
Providers for direct beam
"""
