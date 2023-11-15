# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Loading and masking specific to the ISIS Sans2d instrument and files stored in Scipp's
HDF5 format.
"""
from functools import lru_cache
from typing import Optional

import scipp as sc

from .common import gravity_vector
from .types import (
    BackgroundRun,
    BackgroundTransmissionRun,
    CalibratedMaskedData,
    CleanMasked,
    DataNormalizedByIncidentMonitor,
    DetectorEdgeMask,
    DirectBeam,
    DirectBeamFilename,
    EmptyBeamRun,
    Filename,
    Incident,
    MaskedData,
    MonitorType,
    NeXusMonitorName,
    Numerator,
    RawData,
    RawMonitor,
    RunType,
    SampleHolderMask,
    SampleRun,
    SampleTransmissionRun,
    Transmission,
)


@lru_cache
def pooch_load(filename: Filename[RunType]) -> RawData[RunType]:
    from .data import get_path

    dg = sc.io.load_hdf5(filename=get_path(filename))
    data = dg['data']
    if 'gravity' not in data.coords:
        data.coords["gravity"] = gravity_vector()
    data.coords['pixel_width'] = sc.scalar(0.002033984375, unit='m')
    data.coords['pixel_height'] = sc.scalar(0.0035, unit='m')

    # Some fixes specific for these Sans2d runs
    sample_pos_z_offset = 0.053 * sc.units.m
    # There is some uncertainty here
    monitor4_pos_z_offset = -6.719 * sc.units.m

    data.coords['sample_position'].fields.z += sample_pos_z_offset
    # Results are actually slightly better at high-Q if we do not apply a bench offset
    # bench_pos_y_offset = 0.001 * sc.units.m
    # data.coords['position'].fields.y += bench_pos_y_offset
    dg['monitors']['monitor4']['data'].coords[
        'position'
    ].fields.z += monitor4_pos_z_offset
    return RawData[RunType](dg)


@lru_cache
def pooch_load_direct_beam(filename: DirectBeamFilename) -> DirectBeam:
    from .data import get_path

    return DirectBeam(sc.io.load_hdf5(filename=get_path(filename)))


def _get_monitor(dg: sc.DataGroup, nexus_name: str) -> sc.DataArray:
    # See https://github.com/scipp/sciline/issues/52 why copy needed
    return dg['monitors'][nexus_name]['data'].copy()


def get_empty_beam_incident_monitor(
    dg: RawData[EmptyBeamRun], nexus_name: NeXusMonitorName[Incident]
) -> RawMonitor[EmptyBeamRun, Incident]:
    return RawMonitor[EmptyBeamRun, Incident](_get_monitor(dg, nexus_name))


def get_empty_beam_transmission_monitor(
    dg: RawData[EmptyBeamRun], nexus_name: NeXusMonitorName[Transmission]
) -> RawMonitor[EmptyBeamRun, Transmission]:
    return RawMonitor[EmptyBeamRun, Transmission](_get_monitor(dg, nexus_name))


def get_background_incident_monitor(
    dg: RawData[BackgroundRun], nexus_name: NeXusMonitorName[Incident]
) -> RawMonitor[BackgroundRun, Incident]:
    return RawMonitor[BackgroundRun, Incident](_get_monitor(dg, nexus_name))


def get_background_transmission_monitor(
    dg: RawData[BackgroundRun], nexus_name: NeXusMonitorName[Transmission]
) -> RawMonitor[BackgroundRun, Transmission]:
    return RawMonitor[BackgroundRun, Transmission](_get_monitor(dg, nexus_name))


def get_sample_incident_monitor(
    dg: RawData[SampleRun], nexus_name: NeXusMonitorName[Incident]
) -> RawMonitor[SampleRun, Incident]:
    return RawMonitor[SampleRun, Incident](_get_monitor(dg, nexus_name))


def get_sample_transmission_monitor(
    dg: RawData[SampleTransmissionRun], nexus_name: NeXusMonitorName[Transmission]
) -> RawMonitor[SampleRun, Transmission]:
    return RawMonitor[SampleRun, Transmission](_get_monitor(dg, nexus_name))


def get_sampletransmission_incident_monitor(
    dg: RawData[SampleTransmissionRun], nexus_name: NeXusMonitorName[Incident]
) -> RawMonitor[SampleTransmissionRun, Incident]:
    return RawMonitor[SampleTransmissionRun, Incident](_get_monitor(dg, nexus_name))


def get_backgroundtransmission_incident_monitor(
    da: RawData[BackgroundTransmissionRun], nexus_name: NeXusMonitorName[Incident]
) -> RawMonitor[BackgroundTransmissionRun, Incident]:
    return RawMonitor[BackgroundTransmissionRun, Incident](_get_monitor(da, nexus_name))


def detector_edge_mask(
    raw: DataNormalizedByIncidentMonitor[SampleRun],
) -> DetectorEdgeMask:
    sample = raw['data']
    mask_edges = (
        sc.abs(sample.coords['position'].fields.x) > sc.scalar(0.48, unit='m')
    ) | (sc.abs(sample.coords['position'].fields.y) > sc.scalar(0.45, unit='m'))
    return DetectorEdgeMask(mask_edges)


def sample_holder_mask(
    raw: DataNormalizedByIncidentMonitor[SampleRun],
) -> SampleHolderMask:
    sample = raw['data']
    summed = sample.sum('tof')
    holder_mask = (
        # (summed.data < sc.scalar(100, unit='counts'))
        (summed.data < sc.scalar(4.4e-06))
        & (sample.coords['position'].fields.x > sc.scalar(0, unit='m'))
        & (sample.coords['position'].fields.x < sc.scalar(0.42, unit='m'))
        & (sample.coords['position'].fields.y < sc.scalar(0.05, unit='m'))
        & (sample.coords['position'].fields.y > sc.scalar(-0.15, unit='m'))
    )
    return SampleHolderMask(holder_mask)


def normalize_detector_counts_by_incident_monitor(
    dg: RawData[RunType], incident_monitor: RawMonitor[RunType, Incident]
) -> DataNormalizedByIncidentMonitor[RunType]:
    out = dg.copy(deep=False)
    out['data'] = dg['data'] / sc.values(incident_monitor.data.sum())
    return DataNormalizedByIncidentMonitor[RunType](out)


def mask_detectors(
    dg: DataNormalizedByIncidentMonitor[RunType],
    edge_mask: Optional[DetectorEdgeMask],
    holder_mask: Optional[SampleHolderMask],
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
    da = dg['data'].copy(deep=False)
    if edge_mask is not None:
        da.masks['edges'] = edge_mask
    if holder_mask is not None:
        da.masks['holder_mask'] = holder_mask
    return MaskedData[RunType](da)


def mask_after_calibration(
    da: CalibratedMaskedData[RunType],
) -> CleanMasked[RunType, Numerator]:
    return CleanMasked[RunType, Numerator](da)


providers = [
    pooch_load_direct_beam,
    pooch_load,
    get_empty_beam_incident_monitor,
    get_empty_beam_transmission_monitor,
    get_background_incident_monitor,
    get_background_transmission_monitor,
    get_backgroundtransmission_incident_monitor,
    get_sample_incident_monitor,
    get_sample_transmission_monitor,
    get_sampletransmission_incident_monitor,
    detector_edge_mask,
    sample_holder_mask,
    mask_detectors,
    mask_after_calibration,
    normalize_detector_counts_by_incident_monitor,
]
"""
Providers for loading and masking Sans2d data.

These are meant for complementing the top-level :py:data:`esssans.providers` list.
"""
