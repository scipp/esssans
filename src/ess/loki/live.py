# This file is used by beamlime to create a workflow for the Loki instrument.
# The callable `live_workflow` is registered as the entry point for the workflow.
from pathlib import Path
from typing import NewType

import sciline
import scipp as sc
import scippnexus as snx

from ess import loki
from ess.reduce import streaming
from ess.reduce.nexus import types as nexus_types
from ess.reduce.nexus.json_nexus import JSONGroup
from ess.sans.types import (
    Filename,
    Incident,
    MonitorType,
    NeXusDetectorName,
    NeXusMonitorName,
    RunType,
    SampleRun,
    Transmission,
    WavelengthBins,
    WavelengthMonitor,
    IofQ,
    IofQxy,
    BackgroundSubtractedIofQ,
    BackgroundSubtractedIofQxy,
    CorrectForGravity,
    UncertaintyBroadcastMode,
    ReturnEvents,
    QBins,
    QxBins,
    QyBins,
    BeamCenter,
    DirectBeam,
    BackgroundRun,
    EmptyBeamRun,
    TransmissionRun,
    ReducedQ,
    Numerator,
    Denominator,
    DirectBeamFilename,
)
from ess.sans import with_pixel_mask_filenames
import ess.loki.data  # noqa: F401


class MonitorHistogram(
    sciline.ScopeTwoParams[RunType, MonitorType, sc.DataArray], sc.DataArray
): ...


def _hist_monitor_wavelength(
    wavelength_bin: WavelengthBins, monitor: WavelengthMonitor[RunType, MonitorType]
) -> MonitorHistogram[RunType, MonitorType]:
    return monitor.hist(wavelength=wavelength_bin)


JSONEventData = NewType('JSONEventData', dict[str, JSONGroup])


def load_json_incident_monitor_data(
    name: NeXusMonitorName[Incident],
    nxevent_data: JSONEventData,
) -> nexus_types.NeXusMonitorData[SampleRun, Incident]:
    json = nxevent_data[name]
    group = snx.Group(json, definitions=snx.base_definitions())
    return nexus_types.NeXusMonitorData[SampleRun, Incident](group[()])


def load_json_transmission_monitor_data(
    name: NeXusMonitorName[Transmission],
    nxevent_data: JSONEventData,
) -> nexus_types.NeXusMonitorData[SampleRun, Transmission]:
    json = nxevent_data[name]
    group = snx.Group(json, definitions=snx.base_definitions())
    return nexus_types.NeXusMonitorData[SampleRun, Transmission](group[()])


def load_json_detector_data(
    name: NeXusDetectorName,
    nxevent_data: JSONEventData,
) -> nexus_types.NeXusDetectorData[SampleRun]:
    json = nxevent_data[name]
    group = snx.Group(json, definitions=snx.base_definitions())[()]

    return nexus_types.NeXusDetectorData[SampleRun](group)


class LokiMonitorWorkflow:
    """LoKi Monitor wavelength histogram workflow for live data reduction."""

    def __init__(self, nexus_filename: Path) -> None:
        self._workflow = self._build_pipeline(nexus_filename=nexus_filename)
        self._streamed = streaming.StreamProcessor(
            base_workflow=self._workflow,
            dynamic_keys=(JSONEventData,),
            target_keys=(
                MonitorHistogram[SampleRun, Incident],
                MonitorHistogram[SampleRun, Transmission],
                IofQ[SampleRun],
                IofQxy[SampleRun],
                # BackgroundSubtractedIofQ,
                # BackgroundSubtractedIofQxy,
            ),
            accumulators={
                ReducedQ[SampleRun, Numerator]: streaming.RollingAccumulator(window=20),
                ReducedQ[SampleRun, Denominator]: streaming.RollingAccumulator(
                    window=20
                ),
            },
            # accumulators=(
            #    ReducedQ[SampleRun, Numerator],
            #    ReducedQ[SampleRun, Denominator],
            # ),
        )

    def _build_pipeline(self, nexus_filename: Path) -> sciline.Pipeline:
        """Build a workflow pipeline for live data reduction.

        Returns
        -------
        :
            A pipeline for live data reduction.
            The initial pipeline will be missing the input data.
            It should be set before calling the pipeline.

        """
        workflow = loki.LokiAtLarmorWorkflow()
        workflow = with_pixel_mask_filenames(
            workflow, masks=loki.data.loki_tutorial_mask_filenames()
        )
        workflow[NeXusDetectorName] = 'larmor_detector'
        workflow.insert(_hist_monitor_wavelength)
        workflow.insert(load_json_incident_monitor_data)
        workflow.insert(load_json_transmission_monitor_data)
        workflow.insert(load_json_detector_data)
        workflow[Filename[SampleRun]] = nexus_filename
        workflow[WavelengthBins] = sc.linspace(
            "wavelength", 1.0, 13.0, 200 + 1, unit='angstrom'
        )

        # For IofQ
        workflow[CorrectForGravity] = True
        workflow[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.upper_bound
        workflow[ReturnEvents] = False

        workflow[QBins] = sc.linspace(
            dim='Q', start=0.01, stop=0.3, num=101, unit='1/angstrom'
        )
        workflow[QxBins] = sc.linspace(
            dim='Qx', start=-0.3, stop=0.3, num=61, unit='1/angstrom'
        )
        workflow[QyBins] = sc.linspace(
            dim='Qy', start=-0.3, stop=0.3, num=61, unit='1/angstrom'
        )
        workflow[BeamCenter] = sc.vector(
            value=[-0.02914868, -0.01816138, 0.0], unit='m'
        )

        # workflow[Filename[SampleRun]] = loki.data.loki_tutorial_sample_run_60339()
        # workflow[Filename[TransmissionRun[SampleRun]]] = (
        #     loki.data.loki_tutorial_sample_transmission_run()
        # )

        # AgBeh
        workflow[BeamCenter] = sc.vector(value=[-0.0295995, -0.02203635, 0.0], unit='m')
        # workflow[DirectBeam] = None
        workflow[DirectBeamFilename] = loki.data.loki_tutorial_direct_beam_all_pixels()
        workflow[Filename[SampleRun]] = loki.data.loki_tutorial_agbeh_sample_run()
        # workflow[Filename[BackgroundRun]] = (
        #     loki.data.loki_tutorial_background_run_60393()
        # )

        # TODO The transmission monitor may actually be unused (but gets computed
        # anyway!) if # a separate transmission run is provided. We also need a way to
        # be able to set the # transmission monitor data for the transmission run if
        # such a run is not available.
        workflow[Filename[TransmissionRun[SampleRun]]] = (
            loki.data.loki_tutorial_agbeh_transmission_run()
        )
        # workflow[Filename[TransmissionRun[BackgroundRun]]] = (
        #     loki.data.loki_tutorial_run_60392()
        # )
        workflow[Filename[EmptyBeamRun]] = loki.data.loki_tutorial_run_60392()

        return workflow

    def __call__(
        self, nxevent_data: dict[str, JSONGroup], nxlog: dict[str, JSONGroup]
    ) -> dict[str, sc.DataArray]:
        """

        Returns
        -------
        :
            Plottable Outputs:

            - MonitorHistogram[SampleRun, Incident]
            - MonitorHistogram[SampleRun, Transmission]

        """
        from time import time

        start = time()
        # I think we will be getting the full path, but the workflow only needs the
        # name of the monitor or detector group.
        nxevent_data = {
            key.lstrip('/').split('/')[2]: value for key, value in nxevent_data.items()
        }
        required_keys = {
            self._workflow.compute(NeXusMonitorName[Incident]),
            self._workflow.compute(NeXusMonitorName[Transmission]),
        }
        if not required_keys.issubset(nxevent_data):
            raise ValueError(f"Expected {required_keys}, got {set(nxevent_data)}")
        results = self._streamed.add_chunk({JSONEventData: nxevent_data})
        print(f"Time taken: {time() - start}")
        return {
            'Incident Monitor': results[MonitorHistogram[SampleRun, Incident]],
            'Transmission Monitor': results[MonitorHistogram[SampleRun, Transmission]],
            'I(Q)': results[IofQ[SampleRun]],
            'I(Q_x, Q_y)': results[IofQxy[SampleRun]],
            #'I(Q)': results[BackgroundSubtractedIofQ],
            #'I(Q_x, Q_y)': results[BackgroundSubtractedIofQxy],
        }


# Notes:
# - key to title mapping
# - streaming.TimedAccumulator, since RollingAccumulator is not working as expected if
#   Beamlime is adjusting the window size.
# - Base workflow with options for concrete plugins (background, transmission, ...)
