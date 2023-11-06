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
    CalibratedMaskedData,
    CleanMasked,
    EmptyBeamRun,
    Filename,
    Incident,
    IntegrationTimeNormalizedData,
    IntegrationTimeNormalizedMonitor,
    MaskedData,
    MonitorType,
    NeXusMonitorName,
    Numerator,
    RawData,
    RawMonitor,
    RunType,
    SampleRun,
    SampleRunID,
    SampleTransmissionRun,
    TimeIntegrationNormFactor,
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

    # y_offset = 0.023891 * sc.units.m
    # # z_offset = (25.6100 + 0.0127) * sc.units.m #0.349
    # z_offset = 0.0127 * sc.units.m
    # # Now shift pixels positions to get the correct beam center
    # da.coords['position'].fields.x = (
    #     -da.coords['position'].fields.x * 1.0020211 - 0.0010086124 * sc.units.m
    # )
    # da.coords['position'].fields.y += y_offset
    # da.coords['position'].fields.z += z_offset
    return RawData[RunType](da)


# def load_sample_larmor_run(filename: Filename[SampleRun]) -> UnmergedSampleRawData:
#     return UnmergedSampleRawData(load_larmor_run(filename))


# def load_background_larmor_run(
#     filename: Filename[BackgroundRun],
# ) -> RawData[BackgroundRun]:
#     return RawData[BackgroundRun](load_larmor_run(filename))


# def load_emptybeam_larmor_run(
#     filename: Filename[EmptyBeamRun],
# ) -> RawData[EmptyBeamRun]:
#     return RawData[EmptyBeamRun](load_larmor_run(filename))


# def load_sampletransmission_larmor_run(
#     filename: Filename[SampleTransmissionRun],
# ) -> RawData[SampleTransmissionRun]:
#     return RawData[SampleTransmissionRun](load_larmor_run(filename))


# def _merge_run_events(a, b):
#     out = a.squeeze().bins.concatenate(b.squeeze())
#     for key in a.attrs:
#         if key.startswith('monitor'):
#             out.attrs[key] = sc.scalar(
#                 a.attrs[key].value.bins.concatenate(b.attrs[key].value)
#             )
#     return out


# def merge_sample_runs(
#     runs: sciline.Series[SampleRunID, UnmergedSampleRawData]
# ) -> RawData[SampleRun]:
#     out = reduce(_merge_run_events, runs.values())
#     return RawData[SampleRun](out.bin(tof=1))
#     # input_file1 = f'{data_path}/{runs[0]}-2022-02-28_2215.nxs'
#     # # fixed_file1 = f'{input_file1[:-4]}_fixed.nxs'
#     # data1 = scn.load_nexus(data_file=input_file1)
#     # da1 = data1.squeeze().copy()
#     # summed_data = da1
#     # summed_monitors_1 = da1.attrs['monitor_1'].value
#     # summed_monitors_2 = da1.attrs['monitor_2'].value
#     # # TODO: Should I take tof min and max and use it for setting boundaries

#     # start_tof = data1.coords['tof'][0].values
#     # end_tof = data1.coords['tof'][-1].values
#     # for run in runs[1:]:
#     #     input_file2 = f'{data_path}/{run}-2022-02-28_2215.nxs'
#     #     # fixed_file2 = f'{input_file2[:-4]}_fixed.nxs'
#     #     data2 = scn.load_nexus(data_file=input_file2)
#     #     if start_tof < data2.coords['tof'][0].values:
#     #         start_tof = data1.coords['tof'][0].values
#     #     if end_tof > data2.coords['tof'][-1].values:
#     #         end_tof = data2.coords['tof'][-1].values
#     #     da2 = data2.squeeze().copy()
#     #     summed_data = summed_data.bins.concatenate(da2)
#     #     summed_monitors_1 = summed_monitors_1.bins.concatenate(
#     #         da2.attrs['monitor_1'].value
#     #     )
#     #     summed_monitors_2 = summed_monitors_2.bins.concatenate(
#     #         da2.attrs['monitor_2'].value
#     #     )

#     # edges = sc.linspace('tof', start_tof, end_tof, 2, unit='ns')
#     # summed_binned_data = sc.bin(summed_data, tof=edges)

#     # # Adding montors from first data set
#     # summed_binned_data.attrs['monitor_1'] = data1.attrs['monitor_1']
#     # summed_binned_data.attrs['monitor_2'] = data1.attrs['monitor_2']
#     # return summed_binned_data


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


def get_time_integration_norm_factor(
    incident_monitor: RawMonitor[RunType, Incident],
    reference_monitor: RawMonitor[SampleRun, Incident],
) -> TimeIntegrationNormFactor[RunType]:
    return TimeIntegrationNormFactor[RunType](
        sc.values(incident_monitor.data.sum()) / sc.values(reference_monitor.data.sum())
    )


def normalize_detector_counts_by_time_integration_factor(
    da: RawData[RunType], norm_factor: TimeIntegrationNormFactor[RunType]
) -> IntegrationTimeNormalizedData[RunType]:
    return IntegrationTimeNormalizedData[RunType](da / norm_factor)


def normalize_monitor_counts_by_time_integration_factor(
    monitor: RawMonitor[RunType, MonitorType],
    norm_factor: TimeIntegrationNormFactor[RunType],
) -> IntegrationTimeNormalizedMonitor[RunType, MonitorType]:
    return IntegrationTimeNormalizedMonitor[RunType, MonitorType](monitor / norm_factor)


# def merge_monitor_data(monitors: sciline.Series[SampleRunID, Result]) -> RawMonitor[RunType, MonitorType]:


def to_straws(da: IntegrationTimeNormalizedData[RunType]) -> DataAsStraws[RunType]:
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
) -> DetectorLowCountsStrawMask:
    return DetectorLowCountsStrawMask(
        sample_straws.sum(['tof', 'pixel']).data < sc.scalar(300.0, unit='counts')
    )


# def detector_straw_mask(
#     sample_straws: CalibratedMaskedData[SampleRun],
# ) -> DetectorBadStrawsMask:
#     dims = ['straw']
#     if 'layer' in sample_straws.dims:
#         dims.append('layer')
#     out = sc.zeros(sizes={dim: sample_straws.sizes[dim] for dim in dims}, dtype=bool)
#     out['layer', 0]['straw', 106] = True
#     out['layer', 0]['straw', 107] = True
#     out['layer', 0]['straw', 114] = True


def detector_beam_stop_mask(
    sample_straws: CalibratedMaskedData[SampleRun],
) -> DetectorBeamStopMask:
    pos = sample_straws.coords['position'].copy()
    pos.fields.z *= 0.0
    # x = pos.fields.x
    # y = pos.fields.y
    return DetectorBeamStopMask((sc.norm(pos) < sc.scalar(0.042, unit='m')))
    # return DetectorBeamStopMask(
    #     (abs(x) < sc.scalar(0.03, unit='m')) & (abs(y) < sc.scalar(0.025, unit='m'))
    # )


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
    normalize_detector_counts_by_time_integration_factor,
    normalize_monitor_counts_by_time_integration_factor,
    get_time_integration_norm_factor,
    # load_sample_larmor_run,
    # load_background_larmor_run,
    # load_emptybeam_larmor_run,
    # load_sampletransmission_larmor_run,
    # merge_sample_runs,
]
"""
Providers for direct beam
"""
