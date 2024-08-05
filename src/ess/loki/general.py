# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 202 Scipp contributors (https://github.com/scipp)
"""
Default parameters, providers and utility functions for the loki workflow.
"""

from __future__ import annotations

import sciline
import scipp as sc
from ess import sans
from ess.reduce import nexus
from ess.reduce.workflow import register_workflow
from ess.sans import providers as sans_providers
from ess.sans.parameters import typical_outputs

from ..sans.common import gravity_vector
from ..sans.types import (
    ConfiguredReducibleData,
    ConfiguredReducibleMonitor,
    CorrectForGravity,
    DetectorPixelShape,
    DimsToKeep,
    Incident,
    LabFrameTransform,
    LoadedNeXusDetector,
    LoadedNeXusMonitor,
    MonitorType,
    NeXusDetectorName,
    NeXusMonitorName,
    NonBackgroundWavelengthRange,
    PixelShapePath,
    RawData,
    RawMonitor,
    RawSample,
    RawSource,
    RunType,
    SamplePosition,
    ScatteringRunType,
    SourcePosition,
    TofData,
    TofMonitor,
    TransformationPath,
    Transmission,
    WavelengthBands,
    WavelengthMask,
)
from . import data
from .io import dummy_load_sample


def default_parameters() -> dict:
    return {
        CorrectForGravity: False,
        DimsToKeep: (),
        NeXusMonitorName[Incident]: 'monitor_1',
        NeXusMonitorName[Transmission]: 'monitor_2',
        TransformationPath: 'transform',
        PixelShapePath: 'pixel_shape',
        NonBackgroundWavelengthRange: None,
        WavelengthMask: None,
        WavelengthBands: None,
    }


# problem:
# redundant or missing validation if param does validation but not init of provider input?


# class BinEdges(sc.Variable):
#    def __init__(self, var: sc.Variable, dim: str, unit: str):
#        # check if var has correct dim and unit
#        pass
#
#
# class WavelengthBins(sc.Variable):
#    def __init__(self, var: sc.Variable):
#        super().__init__(var, dim='wavelength', unit='angstrom')


# 1. combo box to select workflow (could have default)
# 2. checkbox + (combo box + list widget) to select desired outputs (defines workflow subgraph) (can have defaults) --> select subset of nodes
# 3. define parameters  (use intersection of param mapping with nodes in graph)
# 4. run workflow, plot, ...


# Open questions:
# - How to return outputs to the user?
# - Want to be able to re-run without re-running the notebook cell that creates the widget (since otherwise params are lost)
# - Soon users will ask for a workspace-list widget, holding outputs from various workflow runs. How to interact with this? See Mantid ADS and WorkspaceProperty.
# - interaction between outputs we want to compute and visibility/mandatory/optional params, what comes first?
#   => selecting outputs is second step of selecting workflow, this then determines which parameters apply

# def LokiIofQWidget():
#     return ess.reduce.make_widget(LokiWorkflow, targets=[IofQ[SampleRun]])


# not like this:
# value = input()
# value = ess.reduce.make_widget(LokiWorkflow)

# Do methods call compute and detector when re-compute is needed?
# widget = ess.reduce.make_widget(LokiWorkflow)
# widget.get()  # dict of outputs

# out = {}
# widget = ess.reduce.make_widget(LokiWorkflow, out=out)


@register_workflow
def LokiAtLarmorWorkflow() -> sciline.Pipeline:
    """
    Workflow with default parameters for Loki test at Larmor.

    This version of the Loki workflow:

    - Uses ISIS XML files to define masks.
    - Sets a dummy sample position [0,0,0] since files do not contain this information.

    Returns
    -------
    :
        Loki workflow as a sciline.Pipeline
    """

    from ess.isissans.io import read_xml_detector_masking

    from . import providers as loki_providers

    params = default_parameters()
    loki_providers = sans_providers + loki_providers

    pipeline = sciline.Pipeline(providers=loki_providers, params=params)
    pipeline.insert(read_xml_detector_masking)
    pipeline[sans.types.NeXusDetectorName] = 'larmor_detector'
    # No sample information in the Loki@Larmor files, use a dummy sample provider
    pipeline.insert(dummy_load_sample)
    pipeline.typical_outputs = typical_outputs
    return pipeline


@register_workflow
def LokiAtLarmorTutorialWorkflow() -> sciline.Pipeline:
    workflow = LokiAtLarmorWorkflow()
    workflow[sans.types.Filename[sans.types.SampleRun]] = (
        data.loki_tutorial_sample_run_60339()
    )
    # TODO This does not work with multiple
    workflow[sans.types.PixelMaskFilename] = data.loki_tutorial_mask_filenames()[0]

    workflow[sans.types.Filename[sans.types.SampleRun]] = (
        data.loki_tutorial_sample_run_60339()
    )
    workflow[sans.types.Filename[sans.types.BackgroundRun]] = (
        data.loki_tutorial_background_run_60393()
    )
    workflow[sans.types.Filename[sans.types.TransmissionRun[sans.types.SampleRun]]] = (
        data.loki_tutorial_sample_transmission_run()
    )
    workflow[
        sans.types.Filename[sans.types.TransmissionRun[sans.types.BackgroundRun]]
    ] = data.loki_tutorial_run_60392()
    workflow[sans.types.Filename[sans.types.EmptyBeamRun]] = (
        data.loki_tutorial_run_60392()
    )
    return workflow


DETECTOR_BANK_RESHAPING = {
    'larmor_detector': lambda x: x.fold(
        dim='detector_number', sizes={'layer': 4, 'tube': 32, 'straw': 7, 'pixel': 512}
    )
}


def get_source_position(
    raw_source: RawSource[RunType],
) -> SourcePosition[RunType]:
    return SourcePosition[RunType](raw_source['position'])


def get_sample_position(
    raw_sample: RawSample[RunType],
) -> SamplePosition[RunType]:
    return SamplePosition[RunType](raw_sample['position'])


def get_detector_data(
    detector: LoadedNeXusDetector[ScatteringRunType],
    detector_name: NeXusDetectorName,
) -> RawData[ScatteringRunType]:
    da = nexus.extract_detector_data(detector)
    if detector_name in DETECTOR_BANK_RESHAPING:
        da = DETECTOR_BANK_RESHAPING[detector_name](da)
    return RawData[ScatteringRunType](da)


def get_monitor_data(
    monitor: LoadedNeXusMonitor[RunType, MonitorType],
) -> RawMonitor[RunType, MonitorType]:
    out = nexus.extract_monitor_data(monitor).copy(deep=False)
    out.coords['position'] = monitor['position']
    return RawMonitor[RunType, MonitorType](out)


def _add_variances_and_coordinates(
    da: sc.DataArray,
    source_position: sc.Variable,
    sample_position: sc.Variable | None = None,
) -> sc.DataArray:
    out = da.copy(deep=False)
    if out.bins is not None:
        content = out.bins.constituents['data']
        if content.variances is None:
            content.variances = content.values
    # Sample position is not needed in the case of a monitor.
    if sample_position is not None:
        out.coords['sample_position'] = sample_position
    out.coords['source_position'] = source_position
    out.coords['gravity'] = gravity_vector()
    return out


def patch_detector_data(
    detector_data: RawData[ScatteringRunType],
    source_position: SourcePosition[ScatteringRunType],
    sample_position: SamplePosition[ScatteringRunType],
) -> ConfiguredReducibleData[ScatteringRunType]:
    return ConfiguredReducibleData[ScatteringRunType](
        _add_variances_and_coordinates(
            da=detector_data,
            source_position=source_position,
            sample_position=sample_position,
        )
    )


def patch_monitor_data(
    monitor_data: RawMonitor[RunType, MonitorType],
    source_position: SourcePosition[RunType],
) -> ConfiguredReducibleMonitor[RunType, MonitorType]:
    return ConfiguredReducibleMonitor[RunType, MonitorType](
        _add_variances_and_coordinates(da=monitor_data, source_position=source_position)
    )


def _convert_to_tof(da: sc.DataArray) -> sc.DataArray:
    da.bins.coords['tof'] = da.bins.coords.pop('event_time_offset')
    if 'event_time_zero' in da.dims:
        da = da.bins.concat('event_time_zero')
    return da


def data_to_tof(
    da: ConfiguredReducibleData[ScatteringRunType],
) -> TofData[ScatteringRunType]:
    return TofData[ScatteringRunType](_convert_to_tof(da))


def monitor_to_tof(
    da: ConfiguredReducibleMonitor[RunType, MonitorType],
) -> TofMonitor[RunType, MonitorType]:
    return TofMonitor[RunType, MonitorType](_convert_to_tof(da))


def detector_pixel_shape(
    detector: LoadedNeXusDetector[ScatteringRunType],
    pixel_shape_path: PixelShapePath,
) -> DetectorPixelShape[ScatteringRunType]:
    return DetectorPixelShape[ScatteringRunType](detector[pixel_shape_path])


def detector_lab_frame_transform(
    detector: LoadedNeXusDetector[ScatteringRunType],
    transform_path: TransformationPath,
) -> LabFrameTransform[ScatteringRunType]:
    return LabFrameTransform[ScatteringRunType](detector[transform_path])


providers = (
    detector_pixel_shape,
    detector_lab_frame_transform,
    get_detector_data,
    get_monitor_data,
    get_sample_position,
    get_source_position,
    patch_detector_data,
    patch_monitor_data,
    data_to_tof,
    monitor_to_tof,
)
