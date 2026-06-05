def pmf(configure, output=None, checkpoint=None):
    """Run potential of mean force calculation.

    Args:
        configure (Path): Configure files containing the input parameters.
        output (Path): Output files containing the output parameters to be reused.
        checkpoint (Path): Checkpoint directory for resumable workflow.
            If provided, enables ai2-kit checkpointing so interrupted
            PMF calculations can be resumed.
    """
    import asyncio
    from catflow.utils.config import load_yaml_configs
    from catflow.tasker.flows.pmf_flow import PMFInput, PMFOutput
    from catflow.tasker.flows.pmf_flow import flow_pmf_calculation

    flow_input = PMFInput.parse_obj(load_yaml_configs(configure))
    if output is not None:
        flow_output = PMFOutput.parse_obj(load_yaml_configs(output))
    else:
        flow_output = PMFOutput(pmf_tasks=[])

    loop = asyncio.get_event_loop()
    loop.run_until_complete(flow_pmf_calculation(
        flow_input, flow_output,
        checkpoint_dir=checkpoint,
    ))
