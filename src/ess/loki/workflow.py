# This file is used by beamlime to create a workflow for the Loki instrument.
# The callable `live_workflow` is registered as the entry point for the workflow.
import sciline
import scipp as sc
import scippnexus as snx
from scippneutron.io.nexus.load_nexus import JSONGroup

from ess.loki.general import (
    get_monitor_data,
    get_source_position,
    monitor_to_tof,
    patch_monitor_data,
)
from ess.loki.io import load_nexus_monitor, load_nexus_source
from ess.sans.conversions import monitor_to_wavelength, sans_monitor
from ess.sans.types import (
    Filename,
    FilePath,
    Incident,
    IofQ,
    MonitorType,
    NeXusMonitorName,
    RunType,
    SampleRun,
    ScatteringRunType,
    Transmission,
    WavelengthBins,
    WavelengthMonitor,
)


class MonitorHistogram(
    sciline.ScopeTwoParams[RunType, MonitorType, sc.DataArray], sc.DataArray
):
    ...


class IofQHistogram(sciline.Scope[ScatteringRunType, sc.DataArray], sc.DataArray):
    ...


def _hist_monitor_wavelength(
    wavelength_bin: WavelengthBins, monitor: WavelengthMonitor[RunType, MonitorType]
) -> MonitorHistogram[RunType, MonitorType]:
    return monitor.hist(wavelength=wavelength_bin)


def _empty_iofq(_: ScatteringRunType) -> IofQ[ScatteringRunType]:
    return IofQ[ScatteringRunType](
        sc.DataArray(
            data=sc.zeros(dims=["Q"], shape=[100]),
            coords={
                "Q": sc.linspace(
                    dim="Q", start=0.01, stop=0.3, num=101, unit="1/angstrom"
                )
            },
        )
    )


def _hist_iofq(i_of_q: IofQ[ScatteringRunType]) -> IofQHistogram[ScatteringRunType]:
    return IofQHistogram[ScatteringRunType](i_of_q.hist("Q"))


class LiveWorkflow:
    """LoKi workflow for live data reduction."""

    def __init__(self) -> None:
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> sciline.Pipeline:
        """Build a workflow pipeline for live data reduction.

        Returns
        -------
        :
            A pipeline for live data reduction.
            The initial pipeline will be missing the input data.
            It should be set before calling the pipeline.

        """
        # Wavelength binning parameters
        wavelength_min = sc.scalar(1.0, unit="angstrom")
        wavelength_max = sc.scalar(13.0, unit="angstrom")
        n_wavelength_bins = 50

        providers = (
            load_nexus_monitor,
            load_nexus_source,
            get_source_position,
            get_monitor_data,
            patch_monitor_data,
            monitor_to_tof,
            sans_monitor,
            monitor_to_wavelength,
            _hist_monitor_wavelength,
            _empty_iofq,
            _hist_iofq,
        )

        params = {
            NeXusMonitorName[Incident]: "monitor_1",
            NeXusMonitorName[Transmission]: "monitor_2",
            WavelengthBins: sc.linspace(
                "wavelength", wavelength_min, wavelength_max, n_wavelength_bins + 1
            ),
        }
        workflow = sciline.Pipeline(providers, params=params)
        return workflow

    def __call__(self, group: JSONGroup) -> dict[str, sc.DataArray]:
        """

        Returns
        -------
        :
            Plottable Outputs:

            - MonitorHistogram[SampleRun, Incident]
            - MonitorHistogram[SampleRun, Transmission]
            - IofQHistogram[SampleRun]

        """
        # ``JsonGroup`` is turned into the ``NexusGroup`` here, not in the ``beamlime``
        # so that the workflow can control the definition of the group.
        self.pipeline[FilePath[Filename[SampleRun]]] = snx.Group(
            group, definitions=snx.base_definitions()
        )
        results = self.pipeline.compute(
            (
                MonitorHistogram[SampleRun, Incident],
                MonitorHistogram[SampleRun, Transmission],
            )
        )

        return {str(tp): result for tp, result in results.items()}


live_workflow = LiveWorkflow()
