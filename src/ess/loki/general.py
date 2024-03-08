# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Default parameters, providers and utility functions for the loki workflow.
"""


from ..sans.types import (
    CalibratedMonitor,
    DetectorPixelShape,
    Incident,
    LabFrameTransform,
    LoadedDetector,
    LoadedMonitor,
    MonitorType,
    NeXusDetectorName,
    NeXusMonitorName,
    RawData,
    RunType,
    ScatteringRunType,
    TransformationPath,
    Transmission,
)

default_parameters = {
    NeXusDetectorName: 'larmor_detector',
    NeXusMonitorName[Incident]: 'monitor_1',
    NeXusMonitorName[Transmission]: 'monitor_2',
    TransformationPath: 'transformation',
}


def get_detector_data(
    dg: LoadedDetector[ScatteringRunType], detector_name: NeXusDetectorName
) -> RawData[ScatteringRunType]:
    return RawData[ScatteringRunType](dg[f'{detector_name}_events'])


def get_monitor_data(
    monitor: LoadedMonitor[RunType, MonitorType],
    monitor_name: NeXusMonitorName[MonitorType],
) -> CalibratedMonitor[RunType, MonitorType]:
    out = monitor[f'{monitor_name}_events'].copy(deep=False)
    out.coords['position'] = monitor['position']
    return CalibratedMonitor[RunType, MonitorType](out)


def detector_pixel_shape(
    dg: LoadedDetector[ScatteringRunType],
) -> DetectorPixelShape[ScatteringRunType]:
    return DetectorPixelShape[ScatteringRunType](dg['pixel_shape'])


def detector_lab_frame_transform(
    detector: LoadedDetector[ScatteringRunType],
    transform_path: TransformationPath,
) -> LabFrameTransform[ScatteringRunType]:
    return LabFrameTransform[ScatteringRunType](detector[transform_path])


providers = (
    detector_pixel_shape,
    detector_lab_frame_transform,
    get_detector_data,
    get_monitor_data,
)
