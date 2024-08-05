# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from collections.abc import Callable, Hashable, Iterable
from typing import Any

import pandas as pd
import sciline
import scipp as sc
from ess.reduce.parameter import Parameter
from ess.reduce.workflow import Workflow
from sciline.typing import Key

from . import parameters
from .types import (
    BackgroundRun,
    BackgroundSubtractedIofQ,
    BackgroundSubtractedIofQxy,
    CleanSummedQ,
    Denominator,
    DetectorMasks,
    Filename,
    Incident,
    IofQ,
    IofQxy,
    MaskedData,
    NeXusDetectorName,
    Numerator,
    PixelMaskFilename,
    SampleRun,
    Transmission,
    WavelengthMonitor,
)

# Plan:
# - Rename to SANSWorkflowInterface, not meant to be inherited from
# - Take pipeline *instance* as argument to __init__
# - Widget workflow selector create workflow, then passes it to interface
# - Workflow selector can be bypassed, if user has created their own pipeline

# What do we actually need:
# - defaults, but could use what was set already on workflow?
#   in fact the widget should do this, no need for double bookkeeping?
# - list of typical outputs; could be stored as attr on workflow?
# - special setters performing map-reduce, store in parameter?

# Other to-dos:
# - auto-gen widgets for parameters not listed by workflow (based on type)


def _get_defaults_from_workflow(workflow: sciline.Pipeline) -> dict[Key, Any]:
    nodes = workflow.underlying_graph.nodes
    return {key: values['value'] for key, values in nodes.items() if 'value' in values}


class SANSWorkflow(Workflow):
    """Base class for SANS workflows, not intended for direct use."""

    @property
    def typical_outputs(self) -> tuple[Key, ...]:
        """Return a tuple of outputs that are used regularly."""

        return (
            BackgroundSubtractedIofQ,
            BackgroundSubtractedIofQxy,
            IofQ[SampleRun],
            IofQxy[SampleRun],
            IofQ[BackgroundRun],
            IofQxy[BackgroundRun],
            MaskedData[BackgroundRun],
            MaskedData[SampleRun],
            WavelengthMonitor[SampleRun, Incident],
            WavelengthMonitor[SampleRun, Transmission],
            WavelengthMonitor[BackgroundRun, Incident],
            WavelengthMonitor[BackgroundRun, Transmission],
        )

    def _parameters(self) -> dict[Key, Parameter]:
        """Return a dictionary of parameters for the workflow."""
        return parameters.make_parameter_mapping(
            defaults=_get_defaults_from_workflow(self.pipeline)
        )

    @property
    def _param_value_setters(
        self,
    ) -> dict[type, Callable[[sciline.Pipeline, Any], sciline.Pipeline]]:
        return {PixelMaskFilename: with_pixel_mask_filenames}


def _merge(*dicts: dict) -> dict:
    return {key: value for d in dicts for key, value in d.items()}


def merge_contributions(*data: sc.DataArray) -> sc.DataArray:
    if len(data) == 1:
        return data[0]
    reducer = sc.reduce(data)
    return reducer.bins.concat() if data[0].bins is not None else reducer.sum()


def with_pixel_mask_filenames(
    workflow: sciline.Pipeline, masks: Iterable[str]
) -> sciline.Pipeline:
    """
    Return modified workflow with pixel mask filenames set.

    Parameters
    ----------
    workflow:
        Workflow to modify.
    masks:
        List or tuple of pixel mask filenames to set.
    """
    workflow = workflow.copy()
    workflow[DetectorMasks] = (
        workflow[DetectorMasks]
        .map(pd.DataFrame({PixelMaskFilename: masks}).rename_axis('mask'))
        .reduce(index='mask', func=_merge)
    )
    return workflow


def with_banks(
    workflow: sciline.Pipeline,
    banks: Iterable[str],
    index: Iterable[Hashable] | None = None,
) -> sciline.Pipeline:
    """
    Return modified workflow with bank names set.

    Since banks typically have different Q-resolution the I(Q) of banks are not merged.
    That is, the resulting workflow will have separate outputs for each bank. Use
    :py:func:`sciline.compute_mapped` to compute results for all banks.

    Parameters
    ----------
    workflow:
        Workflow to modify.
    banks:
        List or tuple of bank names to set.
    index:
        Index to use for the DataFrame. If not provided, the bank names are used.
    """
    index = index or banks
    return workflow.map(
        pd.DataFrame({NeXusDetectorName: banks}, index=index).rename_axis('bank')
    )


def _set_runs(
    pipeline: sciline.Pipeline, runs: Iterable[str], key: Hashable, axis_name: str
) -> sciline.Pipeline:
    pipeline = pipeline.copy()
    runs = pd.DataFrame({Filename[key]: runs}).rename_axis(axis_name)
    for part in (Numerator, Denominator):
        pipeline[CleanSummedQ[key, part]] = (
            pipeline[CleanSummedQ[key, part]]
            .map(runs)
            .reduce(index=axis_name, func=merge_contributions)
        )
    return pipeline


def with_sample_runs(
    workflow: sciline.Pipeline, runs: Iterable[str]
) -> sciline.Pipeline:
    """
    Return modified workflow with sample run filenames set.

    Parameters
    ----------
    workflow:
        Workflow to modify.
    runs:
        List or tuple of sample run filenames to set.
    """
    return _set_runs(workflow, runs, SampleRun, 'sample_run')


def with_background_runs(
    workflow: sciline.Pipeline, runs: Iterable[str]
) -> sciline.Pipeline:
    """
    Return modified workflow with background run filenames set.

    Parameters
    ----------
    workflow:
        Workflow to modify.
    runs:
        List or tuple of background run filenames to set.
    """
    return _set_runs(workflow, runs, BackgroundRun, 'background_run')
