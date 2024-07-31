# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from ess.loki import LokiAtLarmorWorkflow
from ess.sans.types import BackgroundRun, Filename, IofQ, ReturnEvents, SampleRun
from rich import print


def test_loki_workflow_get_parameters():
    workflow = LokiAtLarmorWorkflow()
    parameters = workflow.parameters((IofQ[SampleRun],))
    print(parameters)
    assert False


def test_loki_workflow_parameters_returns_filtered_params():
    workflow = LokiAtLarmorWorkflow()
    parameters = workflow.parameters((IofQ[SampleRun],))
    assert Filename[SampleRun] in parameters
    assert Filename[BackgroundRun] not in parameters


def test_loki_workflow_parameters_returns_no_params_for_no_outputs():
    workflow = LokiAtLarmorWorkflow()
    parameters = workflow.parameters(())
    assert not parameters


def test_loki_workflow_parameters_with_param_returns_param():
    workflow = LokiAtLarmorWorkflow()
    parameters = workflow.parameters((ReturnEvents,))
    assert parameters.keys() == {ReturnEvents}
