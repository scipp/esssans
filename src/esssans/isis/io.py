# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""
File loading functions for ISIS data, NOT using Mantid.
"""
from typing import NewType, TypeVar

import sciline
import scipp as sc

from ..types import DirectBeam, DirectBeamFilename, LoadedFileContents, RunType

PixelMaskFilename = NewType('PixelMaskFilename', str)
CalibrationFilename = NewType('CalibrationFilename', str)

FilenameType = TypeVar('FilenameType', bound=str)


DataFolder = NewType('DataFolder', str)


class FilePath(sciline.Scope[FilenameType, str], str):
    """Path to a file"""


class Filename(sciline.Scope[RunType, str], str):
    """Filename of a run"""


MaskedDetectorIDs = NewType('MaskedDetectorIDs', sc.Variable)
"""1-D variable listing all masked detector IDs."""


def to_path(filename: FilenameType, path: DataFolder) -> FilePath[FilenameType]:
    return f'{path}/{filename}'


def read_xml_detector_masking(
    filename: FilePath[PixelMaskFilename],
) -> MaskedDetectorIDs:
    """Read a pixel mask from an XML file.

    The format is as follows, where the detids are inclusive ranges of detector IDs:

    .. code-block:: xml

        <?xml version="1.0"?>
        <detector-masking>
            <group>
                <detids>1400203-1400218,1401199,1402190-1402223</detids>
            </group>
        </detector-masking>

    Parameters
    ----------
    filename:
        Path to the XML file.
    """
    import xml.etree.ElementTree as ET  # nosec

    tree = ET.parse(filename)  # nosec
    root = tree.getroot()

    masked_detids = []
    for group in root.findall('group'):
        for detids in group.findall('detids'):
            for detid in detids.text.split(','):
                detid = detid.strip()
                if '-' in detid:
                    start, stop = detid.split('-')
                    masked_detids += list(range(int(start), int(stop) + 1))
                else:
                    masked_detids.append(int(detid))

    return MaskedDetectorIDs(
        sc.array(dims=['detector_id'], values=masked_detids, unit=None, dtype='int32')
    )


def load_run(filename: FilePath[Filename[RunType]]) -> LoadedFileContents[RunType]:
    return LoadedFileContents[RunType](sc.io.load_hdf5(filename))


def load_direct_beam(filename: FilePath[DirectBeamFilename]) -> DirectBeam:
    return DirectBeam(sc.io.load_hdf5(filename))


providers = (read_xml_detector_masking, to_path, load_run, load_direct_beam)
