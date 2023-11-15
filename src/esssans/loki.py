# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Loading and masking specific to the ISIS Sans2d instrument and files stored in Scipp's
HDF5 format.
"""
from functools import lru_cache, reduce
from typing import NewType, Optional

import sciline
import scipp as sc
import scippneutron as scn

from .common import gravity_vector
from .types import (
    BackgroundRun,
    BackgroundTransmissionRun,
    CalibratedMaskedData,
    CleanMasked,
    DataNormalizedByIncidentMonitor,
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
    SampleRunID,
    SampleTransmissionRun,
    Transmission,
    UnmergedSampleRawData,
)

DetectorLowCountsStrawMask = NewType('DetectorLowCountsStrawMask', sc.Variable)
"""Detector low-counts straw mask"""
DetectorBadStrawsMask = NewType('DetectorBadStrawsMask', sc.Variable)
"""Detector bad straws mask"""
DetectorBeamStopMask = NewType('DetectorBeamStopMask', sc.Variable)
"""Detector beam stop mask"""
DetectorTubeEdgeMask = NewType('DetectorTubeEdgeMask', sc.Variable)
"""Detector tube edge mask"""


class DataAsStraws(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Data reshaped to have a straw dimension"""


@lru_cache
def load_loki_run(filename: str) -> sc.DataArray:
    from .data import get_path

    # TODO: Use the new scippnexus to avoid using load_nexus, now that transformations
    # are supported.
    da = scn.load_nexus(get_path(filename))
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
    return da


def load_sample_loki_run(filename: Filename[SampleRun]) -> UnmergedSampleRawData:
    return UnmergedSampleRawData(load_loki_run(filename))


def load_background_loki_run(
    filename: Filename[BackgroundRun],
) -> RawData[BackgroundRun]:
    return RawData[BackgroundRun](load_loki_run(filename))


def load_emptybeam_loki_run(
    filename: Filename[EmptyBeamRun],
) -> RawData[EmptyBeamRun]:
    return RawData[EmptyBeamRun](load_loki_run(filename))


def load_sampletransmission_loki_run(
    filename: Filename[SampleTransmissionRun],
) -> RawData[SampleTransmissionRun]:
    return RawData[SampleTransmissionRun](load_loki_run(filename))


def _merge_run_events(a, b):
    out = a.squeeze().bins.concatenate(b.squeeze())
    for key in a.attrs:
        if key.startswith('monitor'):
            out.attrs[key] = sc.scalar(
                a.attrs[key].value.bins.concatenate(b.attrs[key].value)
            )
    return out


def merge_sample_runs(
    runs: sciline.Series[SampleRunID, UnmergedSampleRawData]
) -> RawData[SampleRun]:
    out = reduce(_merge_run_events, runs.values())
    return RawData[SampleRun](out.bin(tof=1))


def _get_monitor(da: sc.DataArray, nexus_name: str) -> sc.DataArray:
    # See https://github.com/scipp/sciline/issues/52 why copy needed
    return da.attrs[nexus_name].value.copy()


def get_empty_beam_incident_monitor(
    da: RawData[EmptyBeamRun], nexus_name: NeXusMonitorName[Incident]
) -> RawMonitor[EmptyBeamRun, Incident]:
    return RawMonitor[EmptyBeamRun, Incident](_get_monitor(da, nexus_name))


def get_empty_beam_transmission_monitor(
    da: RawData[EmptyBeamRun], nexus_name: NeXusMonitorName[Transmission]
) -> RawMonitor[EmptyBeamRun, Transmission]:
    return RawMonitor[EmptyBeamRun, Transmission](_get_monitor(da, nexus_name))


def get_background_incident_monitor(
    da: RawData[BackgroundRun], nexus_name: NeXusMonitorName[Incident]
) -> RawMonitor[BackgroundRun, Incident]:
    return RawMonitor[BackgroundRun, Incident](_get_monitor(da, nexus_name))


def get_background_transmission_monitor(
    da: RawData[BackgroundRun], nexus_name: NeXusMonitorName[Transmission]
) -> RawMonitor[BackgroundRun, Transmission]:
    return RawMonitor[BackgroundRun, Transmission](_get_monitor(da, nexus_name))


def get_sample_incident_monitor(
    da: RawData[SampleRun], nexus_name: NeXusMonitorName[Incident]
) -> RawMonitor[SampleRun, Incident]:
    return RawMonitor[SampleRun, Incident](_get_monitor(da, nexus_name))


def get_sample_transmission_monitor(
    da: RawData[SampleTransmissionRun], nexus_name: NeXusMonitorName[Transmission]
) -> RawMonitor[SampleRun, Transmission]:
    return RawMonitor[SampleRun, Transmission](_get_monitor(da, nexus_name))


def get_sampletransmission_incident_monitor(
    da: RawData[SampleTransmissionRun], nexus_name: NeXusMonitorName[Incident]
) -> RawMonitor[SampleTransmissionRun, Incident]:
    return RawMonitor[SampleTransmissionRun, Incident](_get_monitor(da, nexus_name))


def get_backgroundtransmission_incident_monitor(
    da: RawData[BackgroundTransmissionRun], nexus_name: NeXusMonitorName[Incident]
) -> RawMonitor[BackgroundTransmissionRun, Incident]:
    return RawMonitor[BackgroundTransmissionRun, Incident](_get_monitor(da, nexus_name))


def normalize_detector_counts_by_incident_monitor(
    da: RawData[RunType], incident_monitor: RawMonitor[RunType, Incident]
) -> DataNormalizedByIncidentMonitor[RunType]:
    return DataNormalizedByIncidentMonitor[RunType](
        da / sc.values(incident_monitor.data.sum())
    )


def to_straws(da: DataNormalizedByIncidentMonitor[RunType]) -> DataAsStraws[RunType]:
    return DataAsStraws[RunType](
        da.fold(
            dim='detector_id', sizes=dict(layer=4, tube=32, straw=7, pixel=512)
        ).flatten(dims=['tube', 'straw'], to='straw')
    )


def detector_straw_mask(
    sample_straws: CalibratedMaskedData[SampleRun],
) -> DetectorLowCountsStrawMask:
    return DetectorLowCountsStrawMask(
        # sample_straws.sum(['tof', 'pixel']).data < sc.scalar(300.0, unit='counts')
        sample_straws.sum(['tof', 'pixel']).data
        < sc.scalar(2.5e-5)
    )


def detector_beam_stop_mask(
    sample_straws: CalibratedMaskedData[SampleRun],
) -> DetectorBeamStopMask:
    pos = sample_straws.coords['position'].copy()
    pos.fields.z *= 0.0
    return DetectorBeamStopMask((sc.norm(pos) < sc.scalar(0.042, unit='m')))


def detector_tube_edge_mask(
    sample_straws: CalibratedMaskedData[SampleRun],
) -> DetectorTubeEdgeMask:
    return DetectorTubeEdgeMask(
        (abs(sample_straws.coords['position'].fields.x) > sc.scalar(0.36, unit='m'))
        | (abs(sample_straws.coords['position'].fields.y) > sc.scalar(0.28, unit='m'))
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
    lowcounts_straw_mask: Optional[DetectorLowCountsStrawMask],
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
    if lowcounts_straw_mask is not None:
        da.masks['low_counts'] = lowcounts_straw_mask
    if beam_stop_mask is not None:
        da.masks['beam_stop'] = beam_stop_mask
    if tube_edge_mask is not None:
        da.masks['tube_edges'] = tube_edge_mask
    return CleanMasked[RunType, Numerator](da)


providers = [
    to_straws,
    detector_straw_mask,
    detector_beam_stop_mask,
    detector_tube_edge_mask,
    get_empty_beam_incident_monitor,
    get_empty_beam_transmission_monitor,
    get_background_transmission_monitor,
    get_sample_incident_monitor,
    get_sample_transmission_monitor,
    get_sampletransmission_incident_monitor,
    mask_detectors,
    mask_after_calibration,
    normalize_detector_counts_by_incident_monitor,
    load_background_loki_run,
    load_emptybeam_loki_run,
    load_sample_loki_run,
    load_sampletransmission_loki_run,
    merge_sample_runs,
]
"""
Providers for LoKI
"""
