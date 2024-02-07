# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from typing import NewType, Optional

import scipp as sc

from ..data import Registry
from ..types import LoadedFileContents, MaskedData, RawData, RunType, SampleRun
from .io import DataFolder, FilenameType, FilePath

_registry = Registry(
    instrument='sans2d',
    files={
        # Direct beam file (efficiency of detectors as a function of wavelength)
        'DIRECT_SANS2D_REAR_34327_4m_8mm_16Feb16.dat.h5': 'md5:43f4188301d709aa49df0631d03a67cb',  # noqa: E501
        # Empty beam run (no sample and no sample holder/can)
        'SANS2D00063091.nxs.h5': 'md5:d2a5e59a4489220ecb59d221cb07397e',
        # Sample run (sample and sample holder/can)
        'SANS2D00063114.nxs.h5': 'md5:ffcf9f4c8b1ff02f5f03a569b4930ce1',
        # Background run (no sample, sample holder/can only)
        'SANS2D00063159.nxs.h5': 'md5:92f1da2697818416d6e1497035da5dae',
        # Solid angles of the SANS2D detector pixels computed by Mantid (for tests)
        'SANS2D00063091.SolidAngle_from_mantid.hdf5': 'md5:d57b82db377cb1aea0beac7202713861',  # noqa: E501
    },
    version='1',
)


def get_path(
    filename: FilenameType, folder: Optional[DataFolder]
) -> FilePath[FilenameType]:
    if folder is not None:
        return f'{folder}/{filename}'
    return _registry.get_path(filename)


def get_detector_data(
    dg: LoadedFileContents[RunType],
) -> RawData[RunType]:
    da = dg['data']
    # Remove half of the pixels as the second detector panel is not in the beam path
    return RawData[RunType](da['spectrum', : da.sizes['spectrum'] // 2])


DetectorEdgeMask = NewType('DetectorEdgeMask', sc.Variable)
"""Detector edge mask"""

LowCountThreshold = NewType('LowCountThreshold', sc.Variable)
"""Threshold below which detector pixels should be masked
(low-counts on the edges of the detector panel, and the beam stop)"""

SampleHolderMask = NewType('SampleHolderMask', sc.Variable)
"""Sample holder mask"""


def detector_edge_mask(sample: RawData[SampleRun]) -> DetectorEdgeMask:
    mask_edges = (
        sc.abs(sample.coords['position'].fields.x) > sc.scalar(0.48, unit='m')
    ) | (sc.abs(sample.coords['position'].fields.y) > sc.scalar(0.45, unit='m'))
    return DetectorEdgeMask(mask_edges)


def sample_holder_mask(
    sample: RawData[SampleRun],
    low_counts_threshold: LowCountThreshold,
) -> SampleHolderMask:
    summed = sample.sum('tof')
    holder_mask = (
        (summed.data < low_counts_threshold)
        & (sample.coords['position'].fields.x > sc.scalar(0, unit='m'))
        & (sample.coords['position'].fields.x < sc.scalar(0.42, unit='m'))
        & (sample.coords['position'].fields.y < sc.scalar(0.05, unit='m'))
        & (sample.coords['position'].fields.y > sc.scalar(-0.15, unit='m'))
    )
    return SampleHolderMask(holder_mask)


def mask_detectors(
    da: RawData[RunType],
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
    da = da.copy(deep=False)
    if edge_mask is not None:
        da.masks['edges'] = edge_mask
    if holder_mask is not None:
        da.masks['holder_mask'] = holder_mask
    return MaskedData[RunType](da)


providers = (
    get_path,
    get_detector_data,
    detector_edge_mask,
    sample_holder_mask,
    mask_detectors,
)
