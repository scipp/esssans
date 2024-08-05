# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from typing import NewType

import sciline
import scipp as sc
from ess.reduce.workflow import register_workflow
from ess.sans import providers as sans_providers
from ess.sans.parameters import typical_outputs
from ess.sans.types import MaskedData, SampleRun, ScatteringRunType, TofData

from .data import load_tutorial_direct_beam, load_tutorial_run
from .general import default_parameters
from .mantidio import providers as mantid_providers

DetectorEdgeMask = NewType('DetectorEdgeMask', sc.Variable | None)
"""Detector edge mask"""

LowCountThreshold = NewType('LowCountThreshold', sc.Variable)
"""Threshold below which detector pixels should be masked
(low-counts on the edges of the detector panel, and the beam stop)"""

SampleHolderMask = NewType('SampleHolderMask', sc.Variable | None)
"""Sample holder mask"""


def detector_edge_mask(sample: TofData[SampleRun]) -> DetectorEdgeMask:
    mask_edges = (
        sc.abs(sample.coords['position'].fields.x) > sc.scalar(0.48, unit='m')
    ) | (sc.abs(sample.coords['position'].fields.y) > sc.scalar(0.45, unit='m'))
    return DetectorEdgeMask(mask_edges)


def sample_holder_mask(
    sample: TofData[SampleRun], low_counts_threshold: LowCountThreshold
) -> SampleHolderMask:
    summed = sample.hist()
    holder_mask = (
        (summed.data < low_counts_threshold)
        & (sample.coords['position'].fields.x > sc.scalar(0, unit='m'))
        & (sample.coords['position'].fields.x < sc.scalar(0.42, unit='m'))
        & (sample.coords['position'].fields.y < sc.scalar(0.05, unit='m'))
        & (sample.coords['position'].fields.y > sc.scalar(-0.15, unit='m'))
    )
    return SampleHolderMask(holder_mask)


def mask_detectors(
    da: TofData[ScatteringRunType],
    edge_mask: DetectorEdgeMask,
    holder_mask: SampleHolderMask,
) -> MaskedData[ScatteringRunType]:
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
    if edge_mask is not None:
        da.masks['edges'] = edge_mask
    if holder_mask is not None:
        da.masks['holder_mask'] = holder_mask
    return MaskedData[ScatteringRunType](da)


providers = (detector_edge_mask, sample_holder_mask, mask_detectors)


@register_workflow
def Sans2dWorkflow() -> sciline.Pipeline:
    """Create Sans2d workflow with default parameters."""
    from . import providers as isis_providers

    params = default_parameters()
    sans2d_providers = sans_providers + isis_providers + mantid_providers + providers
    workflow = sciline.Pipeline(providers=sans2d_providers, params=params)
    workflow.typical_outputs = typical_outputs
    return workflow


@register_workflow
def Sans2dTutorialWorkflow() -> sciline.Pipeline:
    """
    Create Sans2d tutorial workflow.

    Equivalent to :func:`Sans2dWorkflow`, but with loaders for tutorial data instead
    of Mantid-based loaders.
    """
    workflow = Sans2dWorkflow()
    workflow.insert(load_tutorial_run)
    workflow.insert(load_tutorial_direct_beam)
    workflow.typical_outputs = typical_outputs
    return workflow
