# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""
File loading functions for ISIS data using Mantid.
"""

import threading
from typing import NewType, NoReturn

import sciline
import scipp as sc
import scippneutron as scn
from scipp.constants import g

from ess.sans.types import DirectBeam, DirectBeamFilename, Filename, RunType, SampleRun

from .io import CalibrationFilename, LoadedFileContents

try:
    import mantid.api as _mantid_api
    import mantid.simpleapi as _mantid_simpleapi
    from mantid.api import MatrixWorkspace
except ModuleNotFoundError:
    # Catch runtime usages of Mantid
    class _MantidFallback:
        def __getattr__(self, name: str) -> NoReturn:
            raise ImportError(
                'Mantid is required to use `sans.isis.mantidio` but is not installed'
            ) from None

    _mantid_api = _MantidFallback()
    _mantid_simpleapi = _MantidFallback
    # Needed for type annotations
    MatrixWorkspace = object

Period = NewType('Period', int | None)
"""Period number of the events."""

CalibrationWorkspace = NewType('CalibrationWorkspace', MatrixWorkspace | None)


class DataWorkspace(sciline.Scope[RunType, MatrixWorkspace], MatrixWorkspace):
    """Workspace containing data"""


def _make_detector_info(ws: MatrixWorkspace) -> sc.DataGroup:
    det_info = scn.mantid.make_detector_info(ws, 'spectrum')
    return sc.DataGroup(det_info.coords)


def _get_detector_ids(ws: DataWorkspace[SampleRun]) -> sc.Variable:
    det_info = _make_detector_info(ws)
    dim = 'spectrum'
    da = sc.DataArray(det_info['detector'], coords={dim: det_info[dim]})
    da = sc.sort(da, dim)  # sort by spectrum index
    if not sc.identical(
        da.coords[dim],
        sc.arange('detector', da.sizes['detector'], dtype='int32', unit=None),
    ):
        raise ValueError("Spectrum-detector mapping is not 1:1, this is not supported.")
    return da.data.rename_dims(detector='spectrum')


def load_calibration(filename: CalibrationFilename) -> CalibrationWorkspace:
    ws = _mantid_simpleapi.Load(Filename=str(filename), StoreInADS=False)
    return CalibrationWorkspace(ws)


def load_direct_beam(filename: DirectBeamFilename) -> DirectBeam:
    dg = scn.load_with_mantid(
        filename=filename,
        mantid_alg="LoadRKH",
        mantid_args={"FirstColumnValue": "Wavelength"},
    )
    da = dg['data']
    del da.coords['spectrum']
    return DirectBeam(da)


def from_data_workspace(
    ws: DataWorkspace[RunType], calibration: CalibrationWorkspace
) -> LoadedFileContents[RunType]:
    if calibration is not None:
        _mantid_simpleapi.CopyInstrumentParameters(
            InputWorkspace=calibration, OutputWorkspace=ws, StoreInADS=False
        )
    up = ws.getInstrument().getReferenceFrame().vecPointingUp()
    dg = scn.from_mantid(ws)
    det_ids = _get_detector_ids(ws)
    # In some instruments (e.g. SANS2D), some pixels are used for other purposes (e.g.
    # live acquisition). They have no detector ids, so we exclude them from the data.
    for dim, shape in det_ids.sizes.items():
        dg['data'] = dg['data'][dim, :shape]
    dg['data'] = dg['data'].squeeze()
    if (dg['data'].bins is not None) and ('tof' in dg['data'].coords):
        del dg['data'].coords['tof']
    dg['data'].coords['detector_id'] = det_ids
    dg['data'].coords['gravity'] = sc.vector(value=-up) * g
    return LoadedFileContents[RunType](dg)


def load_run(filename: Filename[RunType], period: Period) -> DataWorkspace[RunType]:
    # Loading many small files with Mantid is, for some reason, very slow when using
    # the default number of threads in the Dask threaded scheduler (1 thread worked
    # best, 2 is a bit slower but still fast). We can either limit that thread count,
    # or add a lock here, which is more specific.
    with load_run.lock:
        try:
            loaded = _mantid_simpleapi.Load(
                Filename=str(filename), LoadMonitors=True, StoreInADS=False
            )
        except TypeError:
            # Not loaded using LoadEventNexus, so LoadMonitor option is not available.
            loaded = _mantid_simpleapi.Load(Filename=str(filename), StoreInADS=False)
    if isinstance(loaded, _mantid_api.Workspace):
        # A single workspace
        data_ws = loaded
        if isinstance(data_ws, _mantid_api.WorkspaceGroup):
            if period is None:
                raise ValueError(
                    f'Needs {Period} to be set to know what '
                    'section of the event data to load'
                )
            data_ws = data_ws.getItem(period)
    else:
        # Separate data and monitor workspaces
        data_ws = loaded.OutputWorkspace
        if isinstance(data_ws, _mantid_api.WorkspaceGroup):
            if period is None:
                raise ValueError(
                    f'Needs {Period} to be set to know what '
                    'section of the event data to load'
                )
            data_ws = data_ws.getItem(period)
            data_ws.setMonitorWorkspace(loaded.MonitorWorkspace.getItem(period))
        else:
            data_ws.setMonitorWorkspace(loaded.MonitorWorkspace)
    return DataWorkspace[RunType](data_ws)


load_run.lock = threading.Lock()

providers = (
    from_data_workspace,
    load_calibration,
    load_direct_beam,
    load_run,
)
