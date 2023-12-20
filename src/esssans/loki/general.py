# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Default parameters, providers and utility functions for the loki workflow.
"""

from collections.abc import Iterable
from itertools import groupby

import sciline
import scipp as sc

from ..common import gravity_vector
from ..types import (
    DataWithLogicalDims,
    DetectorPixelShape,
    Incident,
    LabFrameTransform,
    LoadedFileContents,
    # MonitorType,
    NexusDetectorName,
    NexusInstrumentPath,
    NeXusMonitorName,
    NexusSourceName,
    RawData,
    # RawMonitor,
    # RunID,
    RunType,
    # SamplePosition,
    # SourcePosition,
    # TofData,
    # TofMonitor,
    TransformationChainPath,
    Transmission,
    # UnmergedPatchedData,
    # UnmergedPatchedMonitor,
    # UnmergedRawData,
    # UnmergedRawMonitor,
)


default_parameters = {
    NexusInstrumentPath: 'instrument',
    NexusDetectorName: 'larmor_detector',
    NeXusMonitorName[Incident]: 'monitor_1',
    NeXusMonitorName[Transmission]: 'monitor_2',
    NexusSourceName: 'source',
    # TODO: sample is not in the files, so by not adding the name here, we use the
    # default value of [0, 0, 0] when loading the sample position.
    TransformationChainPath: 'transformation_chain',
}


# def _patch_data(
#     da: sc.DataArray,
#     sample_position: sc.Variable,
#     source_position: sc.Variable,
# ) -> sc.DataArray:
#     out = da.copy(deep=False)
#     out.coords['sample_position'] = sample_position
#     out.coords['source_position'] = source_position
#     out.coords['gravity'] = gravity_vector()
#     return out


# def patch_detector_data(
#     da: UnmergedRawData[RunType],
#     sample_position: SamplePosition[RunType],
#     source_position: SourcePosition[RunType],
# ) -> UnmergedPatchedData[RunType]:
#     return UnmergedPatchedData[RunType](
#         _patch_data(
#             da=da, sample_position=sample_position, source_position=source_position
#         )
#     )


# def patch_monitor_data(
#     da: UnmergedRawMonitor[RunType, MonitorType],
#     sample_position: SamplePosition[RunType],
#     source_position: SourcePosition[RunType],
# ) -> UnmergedPatchedMonitor[RunType, MonitorType]:
#     return UnmergedPatchedMonitor[RunType, MonitorType](
#         _patch_data(
#             da=da, sample_position=sample_position, source_position=source_position
#         )
#     )


# def _convert_to_tof(da: sc.DataArray) -> sc.DataArray:
#     out = da.copy(deep=False)
#     out.bins.coords['tof'] = out.bins.coords.pop('event_time_offset')
#     if 'event_time_zero' in out.dims:
#         out = out.bins.concat('event_time_zero')
#     return out.bin(tof=1)


# def convert_detector_to_tof(
#     da: RawData[RunType],
# ) -> TofData[RunType]:
#     # TODO: This is where the frame unwrapping would occur
#     return TofData[RunType](_convert_to_tof(da))


# def convert_monitor_to_tof(
#     da: RawMonitor[RunType, MonitorType],
# ) -> TofMonitor[RunType, MonitorType]:
#     return TofMonitor[RunType, MonitorType](_convert_to_tof(da))


def to_logical_dims(da: RawData[RunType]) -> DataWithLogicalDims[RunType]:
    return DataWithLogicalDims[RunType](
        da.fold(
            dim='detector_number', sizes=dict(layer=4, tube=32, straw=7, pixel=512)
        ).flatten(dims=['tube', 'straw'], to='straw')
    )


# def _all_equal(iterable: Iterable[sc.Variable]) -> bool:
#     """
#     Returns True if all the elements are equal to each other.
#     From https://stackoverflow.com/a/3844832.
#     """
#     g = groupby(iterable)
#     return next(g, True) and not next(g, False)


def detector_pixel_shape(
    dg: LoadedFileContents[RunType],
    instrument_path: NexusInstrumentPath,
    detector_name: NexusDetectorName,
) -> DetectorPixelShape[RunType]:
    # pixel_shapes = (dg['pixel_shape'] for dg in data_groups.values())
    # if not _all_equal(pixel_shapes):
    #     raise RuntimeError(f'All pixel shapes must be the same for {RunType}')
    return DetectorPixelShape[RunType](
        dg[instrument_path][detector_name]['pixel_shape']
    )


def detector_lab_frame_transform(
    dg: LoadedFileContents[RunType],
    instrument_path: NexusInstrumentPath,
    detector_name: NexusDetectorName,
    transform_path: TransformationChainPath,
) -> LabFrameTransform[RunType]:
    # transforms = (dg[transform_path] for dg in data_groups.values())
    # if not _all_equal(transforms):
    #     raise RuntimeError(f'All lab frame transforms must be the same for {RunType}')
    return LabFrameTransform[RunType](
        dg[instrument_path][detector_name][transform_path]
    )


# providers = tuple()
providers = (
    detector_pixel_shape,
    detector_lab_frame_transform,
    to_logical_dims,
    # patch_detector_data,
    # patch_monitor_data,
    # convert_detector_to_tof,
    # convert_monitor_to_tof,
)
