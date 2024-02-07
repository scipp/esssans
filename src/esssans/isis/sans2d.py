# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)


from ..data import Registry
from ..types import LoadedFileContents, RawData, RunType

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

get_path = _registry.get_path


def get_detector_data(
    dg: LoadedFileContents[RunType],
) -> RawData[RunType]:
    da = dg['data']
    # Remove half of the pixels as the second detector panel is not in the beam path
    return RawData[RunType](da['spectrum', : da.sizes['spectrum'] // 2])


providers = (get_path, get_detector_data)
