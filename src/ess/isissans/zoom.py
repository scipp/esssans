# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from typing import Any

import sciline
from ess.reduce.workflow import register_workflow
from ess.sans import providers as sans_providers
from ess.sans.workflow import SANSWorkflow

from .data import load_tutorial_direct_beam, load_tutorial_run
from .general import default_parameters
from .io import read_xml_detector_masking
from .mantidio import providers as mantid_providers


def set_mantid_log_level(level: int = 3):
    try:
        from mantid import ConfigService

        cfg = ConfigService.Instance()
        cfg.setLogLevel(level)  # Silence verbose load via Mantid
    except ImportError:
        pass


@register_workflow
class ZoomWorkflow(SANSWorkflow):
    """Create Zoom workflow with default parameters."""

    def __init__(self):
        from . import providers as isis_providers

        set_mantid_log_level()

        params = default_parameters()
        zoom_providers = sans_providers + isis_providers + mantid_providers
        pipeline = sciline.Pipeline(providers=zoom_providers, params=params)
        pipeline.insert(read_xml_detector_masking)
        super().__init__(pipeline)

    def _default_param_values(self) -> dict[sciline.typing.Key, Any]:
        return default_parameters()


@register_workflow
class ZoomTutorialWorkflow(ZoomWorkflow):
    """
    Create Zoom tutorial workflow.

    Equivalent to :func:`ZoomWorkflow`, but with loaders for tutorial data instead
    of Mantid-based loaders.
    """

    def __init__(self):
        super().__init__()
        self.pipeline.insert(load_tutorial_run)
        self.pipeline.insert(load_tutorial_direct_beam)
