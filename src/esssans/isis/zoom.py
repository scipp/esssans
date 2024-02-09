# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from typing import NewType, Optional

import numpy as np
import sciline
import scipp as sc


from ..data import Registry
from ..types import LoadedFileContents, MaskedData, RawData, RunType, SampleRun
from .components import RawDataWithComponentUserOffsets
from .io import DataFolder, FilenameType, FilePath, MaskedDetectorIDs


_registry = Registry(
    instrument='zoom',
    files={
        # Sample run (sample and sample holder/can) with applied 192tubeCalibration_11-02-2019_r5_10lines.nxs  # noqa: E501
        'ZOOM00034786.nxs.h5.zip': 'md5:e1c53bf826dd87545df1b3629f424762',
        # Empty beam run (no sample and no sample holder/can) - Scipp-hdf5 format
        'ZOOM00034787.nxs.h5': 'md5:27e563d4e57621518658307acbbc3413',
        # Calibration file, Mantid processed NeXus
        '192tubeCalibration_11-02-2019_r5_10lines.nxs': 'md5:ca1e0e3c387903be445d0dfd0a784ed6',  # noqa: E501
        # Direct beam file (efficiency of detectors as a function of wavelength)
        'Direct_Zoom_4m_8mm_100522.txt.h5': 'md5:bbe813580676a9ad170934ffb7c99617',
        # Moderator file (used for computing Q-resolution)
        'ModeratorStdDev_TS2_SANS_LETexptl_07Aug2015.txt': 'md5:5fc389340d453b9095a5dfcc33608dae',  # noqa: E501
        # ISIS user file configuring the data reduction
        'USER_ZOOM_Cabral_4m_TJump_233G_8x8mm_Small_BEAMSTOP_v1_M5.toml': 'md5:4423ecb7d924c79711aba5b0a30a23e7',  # noqa: E501
        # 7 pixel mask files for the ZOOM00034786.nxs run
        'andru_test.xml': 'md5:c59e0c4a80640a387df7beca4857e66f',
        'left_beg_18_2.xml': 'md5:5b24a8954d4d8a291f59f5392cd61681',
        'right_beg_18_2.xml': 'md5:fae95a5056e5f5ba4996c8dff83ec109',
        'small_bs_232.xml': 'md5:6d67dea9208193c9f0753ffcbb50ed83',
        'small_BS_31032023.xml': 'md5:3c644e8c75105809ab521773f9c0c85b',
        'tube_1120_bottom.xml': 'md5:fe577bf73c16bf5ac909516fa67360e9',
        'tubes_beg_18_2.xml': 'md5:2debde8d82c383cc3d592ea000552300',
    },
    version='2',
)


def get_path(
    filename: FilenameType, folder: Optional[DataFolder]
) -> FilePath[FilenameType]:
    """Translate any filename to a path to the file obtained from pooch registry."""
    if folder is not None:
        return f'{folder}/{filename}'
    mapping = {
        'Direct_Zoom_4m_8mm_100522.txt': 'Direct_Zoom_4m_8mm_100522.txt.h5',
        'ZOOM00034786.nxs': 'ZOOM00034786.nxs.h5.zip',
        'ZOOM00034787.nxs': 'ZOOM00034787.nxs.h5',
    }
    filename = mapping.get(filename, filename)
    if filename.endswith('.zip'):
        return _registry.get_path(filename, unzip=True)[0]
    return _registry.get_path(filename)


def get_detector_data(
    dg: LoadedFileContents[RunType],
) -> RawData[RunType]:
    return RawData[RunType](dg['data'])


providers = (get_path, get_detector_data)
