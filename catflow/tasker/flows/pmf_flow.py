import asyncio
import itertools
import numpy as np

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Literal, Union, List, Tuple, Optional
from pathlib import Path
from MDAnalysis import Universe
from pydantic import BaseModel

from catflow.utils import logger
from catflow.analyzer.structure.cluster import Cluster
from catflow.tasker.tasks.pmf import PMFTask, DPPMFTask
from catflow.tasker.resources.workflow_executor import WorkflowExecutor
from catflow.tasker.resources.config import MachineConfig, SbatchResources
from catflow.utils.config import load_yaml_config, get_item_from_list
from catflow.core import apply_checkpoint, set_checkpoint_dir


class JobConfig(BaseModel):
    """Job configuration."""
    work_path: str
    machine_name: str
    resources: dict
    command: str
    job_name: Optional[str] = None


class PMFJobConfig(BaseModel):
    """Job configuration."""
    work_path: str
    machine_name: str
    resources: dict
    command: str
    reaction_pair: List[int]
    job_name: str = "PMF"
    input_dict: Optional[dict] = None
    cell: Optional[List[float]] = None
    steps: Optional[int] = None
    timestep: Optional[float] = None
    dump_freq: Optional[int] = None
    restart_steps: Optional[int] = None
    outlog: Optional[str] = "output"
    errlog: Optional[str] = "error"
    forward_common_files: Optional[List[str]] = None
    backward_common_files: Optional[List[str]] = None
    forward_files: Optional[List[str]] = None
    backward_files: Optional[List[str]] = None
    model_path: Optional[str] = None
    type_map: Optional[dict] = None


class PMFFlowInitArtifact(BaseModel):
    """Potential of mean force calculation configuration."""
    coordinate: float
    structure_path: str


class PMFFlowConfig(BaseModel):
    """Potential of mean force calculation configuration."""
    coordinates: List[float]
    t_step: float = 100.
    n_temps: int = 5
    lindemann_n_last_frames: int = 20000
    melting_test: bool = True
    temperatures: Optional[List[float]] = None
    t_min: Optional[float] = None
    t_max: Optional[float] = None
    init_artifact: Optional[List[PMFFlowInitArtifact]] = None
    is_coordinate: Optional[float] = None
    fs_coordinate: Optional[float] = None
    ts_coordinate: Optional[float] = None
    cluster_component: Optional[List[str]] = None


class PMFInput(BaseModel):
    """Potential of mean force calculation configuration."""
    job_config: PMFJobConfig
    flow_config: PMFFlowConfig
    dump_dict: Optional[dict] = None
    job_type: Literal["pmf", "dp_pmf"] = "pmf"


class PMFTaskOutput(BaseModel):
    """Potential of mean force calculation output."""
    coordinate: float
    temperature: float
    task_path: str
    restart_time: int
    last_frame_path: Optional[str] = None
    convergence: bool = False
    lindemann_index: Optional[float] = None
    pmf_mean: Optional[float] = None
    pmf_var: Optional[float] = None

    def __hash__(self):
        return hash((
            self.coordinate,
            self.temperature
        ))


class PMFOutput(BaseModel):
    """Potential of mean force calculation output."""
    pmf_tasks: List[PMFTaskOutput]
    temperature_range: List[float] = []


class SafeList:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.items: List[PMFTaskOutput] = []

    async def get_items(self):
        async with self.lock:
            return list(self.items)

    async def add_item(self, new_item: PMFTaskOutput):
        async with self.lock:
            old_item = next(
                (item for item in self.items if hash(item) == hash(new_item)),
                None
            )
            if old_item is not None:
                if (new_item.restart_time > old_item.restart_time) or (
                        new_item.last_frame_path != old_item.last_frame_path) or (
                        new_item.lindemann_index != old_item.lindemann_index) or (
                        new_item.pmf_mean != old_item.pmf_mean) or (
                        new_item.convergence != old_item.convergence):
                    self.items.remove(old_item)
                    self.items.append(new_item)
                else:
                    return
            else:
                self.items.append(new_item)

    async def add_items(self, new_items: List[PMFTaskOutput]):
        for item in new_items:
            await self.add_item(item)


async def regularly_dump_pmf_output(
    flow_input: PMFInput,
    flow_output: PMFOutput,
    pmf_task_outputs: SafeList
) -> None:
    """Regularly dump the PMF output."""
    while True:
        await dump_pmf_output(flow_input, flow_output, pmf_task_outputs)
        await asyncio.sleep(60)


async def dump_pmf_output(
    flow_input: PMFInput,
    flow_output: PMFOutput,
    pmf_task_outputs: SafeList
) -> PMFOutput:
    """Dump the PMF output to a file."""

    from catflow.utils.config import dump_yaml_config

    work_path = Path(flow_input.job_config.work_path).resolve()
    flow_output.pmf_tasks = await pmf_task_outputs.get_items()
    dump_yaml_config(work_path / "pmf_flow_output.yaml", flow_output)
    return flow_output


async def flow_pmf_calculation(
    flow_input: PMFInput,
    flow_output: Optional[PMFOutput] = None,
    checkpoint_dir: Optional[str] = None,
) -> PMFOutput:
    """Calculate the PMF of a given COLVAR.

    Args:
        flow_input: PMF calculation configuration.
        flow_output: Optional existing output state for resuming.
        checkpoint_dir: Directory for ai2-kit checkpoint storage.
            Enables resumable workflow across interruptions.
    """

    # Enable checkpointing for resumable workflow
    if checkpoint_dir is not None:
        set_checkpoint_dir(checkpoint_dir)
        logger.info(f"Checkpoint enabled at: {checkpoint_dir}")

    # define list to store the tasks
    pmf_task_outputs = SafeList()
    if flow_output is None:
        flow_output = PMFOutput(pmf_tasks=[])
    else:
        for task in flow_output.pmf_tasks:
            await pmf_task_outputs.add_item(task)

    dump_task = asyncio.create_task(regularly_dump_pmf_output(
        flow_input, flow_output, pmf_task_outputs
    ))

    # determine the temperature range of the PMF calculation
    if not flow_input.flow_config.melting_test and flow_input.flow_config.temperatures:
        # skip melting test if `melting_test` is set to False and `temperatures` is given
        flow_output.temperature_range = flow_input.flow_config.temperatures
    else:
        if flow_output.temperature_range:
            pass
        else:
            temperature_range = \
                await subflow_pmf_temperature_range(flow_input, pmf_task_outputs)
            flow_output.temperature_range = temperature_range

    # define PMF workflow at each temperature
    pmf_tasks = []
    for temperature in flow_output.temperature_range:
        pmf_tasks.append(asyncio.ensure_future(
            subflow_pmf_each_temperauture(
                temperature, flow_input, pmf_task_outputs
            )))

    await asyncio.gather(*pmf_tasks)

    # dump output to a file
    dump_task.cancel()

    await dump_pmf_output(flow_input, flow_output, pmf_task_outputs)

    # plot the PMF curve
    plot_pmf_profile(flow_input, flow_output)

    return flow_output


@apply_checkpoint(lambda coordinate, temperature, **kw: f"pmf_task_{coordinate}_{temperature}")
async def task_pmf_calculation(
    coordinate: float,
    temperature: float,
    flow_input: PMFInput,
    pmf_task_outputs: SafeList,
    init_structure_path: Optional[str] = None,
    restart_time: int = 0
) -> PMFTaskOutput:
    """Run potential of mean force calculation.

    Checkpointed by (coordinate, temperature) pair so completed tasks
    are skipped on resume.
    """
    logger.info(f"Running PMF calculation at {temperature} K.")
    logger.info(f"Coordinate: {coordinate}")

    pmf_job_config = flow_input.job_config.dict()
    if init_structure_path is None:
        artifact = get_item_from_list(
            flow_input.flow_config.init_artifact,
            coordinate=coordinate
        )
        if artifact:
            _init_structure_path = artifact[1].structure_path # type: str
        else:
            raise ValueError("No initial structure found.")
    else:
        _init_structure_path = init_structure_path
    if flow_input.job_type == "pmf":
        task_run = PMFTask(
            coordinate=coordinate,
            temperature=temperature,
            init_structure_path=_init_structure_path,
            restart_time=restart_time,
            **pmf_job_config
        )
    elif flow_input.job_type == "dp_pmf":
        task_run = DPPMFTask(
            coordinate=coordinate,
            temperature=temperature,
            init_structure_path=_init_structure_path,
            restart_time=restart_time,
            **pmf_job_config
        )
    else:
        raise ValueError("`job_type` should be `pmf` or `dp_pmf`")

    work_path = Path(flow_input.job_config.work_path).resolve()
    task_path = str(
        work_path / f'task.{coordinate}_{temperature}'
    )

    task_output = PMFTaskOutput(
        coordinate=coordinate,
        temperature=temperature,
        task_path=task_path,
        restart_time=restart_time
    )

    await pmf_task_outputs.add_item(task_output)

    # check if the task is finished
    task_finished_tag = task_run.task_path / "task_finished.tag"
    if task_finished_tag.exists():
        logger.info(f"Task {task_run.task_path} finished, skipped.")
    else:
        # New path: generate oh-my-batch script and execute via WorkflowExecutor
        script_path = task_run.generate_script()
        executor = WorkflowExecutor.local(work_base=str(task_run.work_path))
        result = await asyncio.to_thread(
            executor.run_script, script_path, str(task_run.work_path)
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"PMF task failed at {task_run.task_path}. "
                f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )
        logger.info(f"Task {task_run.task_path} finished.")
        task_finished_tag.touch()

    # last frame_path
    last_frame_path = str(task_run.get_last_frame())
    task_output.last_frame_path = last_frame_path
    await pmf_task_outputs.add_item(task_output)

    # task only return task_output
    return task_output


async def convergence_test_lagrange_multiplier(
    task_output: PMFTaskOutput,
    pmf_task_outputs: SafeList
):
    """Convergence test for lagrange multiplier."""
    coordinate = task_output.coordinate
    temperature = task_output.temperature
    task_path = Path(task_output.task_path).resolve()
    restart_time = task_output.restart_time
    with ProcessPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        _partial = partial(
            pmf_analyzer, 
            coordinate, 
            temperature, 
            task_path, 
            restart_time
        )
        mean, var = await loop.run_in_executor(
            executor, _partial
        )
    if var < 0.1:
        # set threshold to 0.1
        logger.info(f"Convergence reached for {task_output.coordinate} at {task_output.temperature}.")
        logger.info(f"Mean: {mean}, Var: {var}")
        task_output.convergence = True
        task_output.pmf_mean = mean
        task_output.pmf_var = var
    else:
        logger.info(f"Convergence not reached for {task_output.coordinate} at {task_output.temperature}.")
        logger.info(f"Mean: {mean}, Var: {var}")
        task_output.convergence = False
        task_output.pmf_mean = mean
        task_output.pmf_var = var
    await pmf_task_outputs.add_item(task_output)
    return task_output


def pmf_analyzer(
    coordinate: float,
    temperature: float,
    task_path: Path,
    restart_time: int = 0
):
    from catflow.utils.cp2k import lagrange_mult_log_parser
    from catflow.utils.statistics import block_average

    logger.info(f"Calculating PMF for {coordinate} at {temperature} K.")

    parsed_log_path = task_path / "lagrange_mult_log_parsed.npy"
    if parsed_log_path.exists():
        lagrange_mults = list(np.load(parsed_log_path))
    else:
        lagrange_mult_log_path = task_path / "pmf.LagrangeMultLog"
        lagrange_mults = lagrange_mult_log_parser(lagrange_mult_log_path)
        np.save(parsed_log_path, lagrange_mults)
    if restart_time > 0:
        for i in range(restart_time):
            _task_path = \
                task_path.parent / f"restart.{i}.{coordinate}_{temperature}"
            parsed_log_path = _task_path / "lagrange_mult_log_parsed.npy"
            if parsed_log_path.exists():
                lagrange_mults += list(np.load(parsed_log_path))
            else:
                lagrange_mult_log_path = _task_path / "pmf.LagrangeMultLog"
                new_lagrange_mults = lagrange_mult_log_parser(lagrange_mult_log_path)
                np.save(parsed_log_path, new_lagrange_mults)
                lagrange_mults += new_lagrange_mults
    mean, var = block_average(
        lagrange_mults[1:], int(len(lagrange_mults[1:])/10)
    )
    return mean, var


async def convergence_test_energy(
    task_output: PMFTaskOutput,
    pmf_task_outputs: SafeList,
):
    pass


def lindemann_index_calculation(
    flow_input: PMFInput,
    task_output: PMFTaskOutput
) -> List[float]:
    """Calculate Lindemann index. Must be followed by PMFCalcualtion.

    Args:
        coordinate (_type_): _description_
        temperature (_type_): _description_
        flow_input (PMFInput): _description_
        flow_output (PMFOutput): _description_

    Raises:
        ValueError: _description_

    Returns:
        float: _description_
    """
    coordinate = task_output.coordinate
    temperature = task_output.temperature
    task_path = Path(task_output.task_path)
    restart_time = task_output.restart_time

    trajectory_path = task_path / "pmf-pos-1.xyz"
    if restart_time > 0:
        restart_list = [trajectory_path]
        restart_list += [task_path.parent /
                         f"restart.{i}.{coordinate}_{temperature}/pmf-pos-1.xyz"
                         for i in range(restart_time)]
        u = Universe(task_path / "pmf-pos-1.xyz", *restart_list)
        cluster = Cluster.convert_universe(u)
    else:
        cluster = Cluster(trajectory_path)
    if flow_input.flow_config.cluster_component is not None:
        name_of_each = [
            "name "+ele for ele in flow_input.flow_config.cluster_component
        ]
        sele_lang = " or ".join(name_of_each)
    else:
        sele_lang = "name all"
    lindemann_now = cluster.lindemann_per_frames(sele_lang)
    # dump parsed lindemann index to file
    np.save(task_path / "lindemann_index.npy", lindemann_now)
    lindemann_indexes = np.mean(lindemann_now, axis=1)
    return list(lindemann_indexes)


async def convergence_test_lindemann_index(
    flow_input: PMFInput,
    task_output: PMFTaskOutput,
    pmf_task_outputs: SafeList,
    n_last_frames: int = 2000000
) -> PMFTaskOutput:
    """Convergence test for Lindemann index."""
    coordinate = task_output.coordinate
    temperature = task_output.temperature
    logger.info(
        f"Calculating Lindemann index for {coordinate} at {temperature} K.")
    with ProcessPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        _partial_ldx_calc = partial(
            lindemann_index_calculation, flow_input, task_output
        )
        lindemann_indexes = await loop.run_in_executor(
            executor, _partial_ldx_calc
        )
    # use last 2000000 frames to calculate
    last_frame_index = lindemann_indexes[-1]
    last_indexes_mean = np.mean(lindemann_indexes[-n_last_frames:])
    scale = np.abs(
        (last_frame_index - last_indexes_mean) / last_indexes_mean
    )
    if scale < 0.01:
        logger.info(f"Convergence reached for {coordinate} at {temperature}.")
        logger.info(f"Last frame index: {last_frame_index}")
        task_output.convergence = True
        task_output.lindemann_index = last_frame_index
    else:
        logger.info(f"Convergence not reached for {coordinate} at {temperature}.")
        logger.info(f"Last frame index: {last_frame_index}")
        task_output.convergence = False
        task_output.lindemann_index = last_frame_index
    await pmf_task_outputs.add_item(task_output)
    return task_output


async def subflow_melting_coordinate(
    coordinate: float,
    temperatures: List[float],
    flow_input: PMFInput,
    pmf_task_outputs: SafeList
) -> List[PMFTaskOutput]:
    """Calculate melting point at given coordinate.

    Args:
        coordinate (float): _description_
        temperatures (List[float]): _description_
        flow_input (PMFInput): _description_
        flow_output (PMFOutput): _description_

    Returns:
        List[PMFTaskOutput]: _description_
    """
    task_outputs = await pmf_task_outputs.get_items()

    pmf_tasks = []
    task_indexes = []
    max_index = len(task_outputs)
    for i, temperature in enumerate(temperatures):
        task_output = None
        task_index = None
        task_check = get_item_from_list(
            task_outputs,
            coordinate=coordinate,
            temperature=temperature
        )
        if task_check is not None:
            task_index = task_check[0]
            task_output = task_check[1]
            if task_output.convergence:
                if task_output.lindemann_index is not None:
                    continue
        # check if the previous PMF calculation is done
        # if so, skip this calculation
        if task_output:
            pmf_tasks.append(
                asyncio.ensure_future(subflow_melting_temperature(
                    coordinate,
                    temperature,
                    flow_input,
                    pmf_task_outputs,
                    restart_time=task_output.restart_time,
                    task_output=task_output
                ))
            )
        else:
            pmf_tasks.append(
                asyncio.ensure_future(subflow_melting_temperature(
                    coordinate,
                    temperature,
                    flow_input,
                    pmf_task_outputs,
                ))
            )
        if task_index:
            task_indexes.append(task_index)
        else:
            task_indexes.append(max_index)

    await asyncio.gather(*pmf_tasks)
    return task_outputs


async def subflow_melting_temperature(
    coordinate: float,
    temperature: float,
    flow_input: PMFInput,
    pmf_task_outputs: SafeList,
    restart_time: int = 0,
    task_output: Optional[PMFTaskOutput] = None
) -> PMFTaskOutput:
    """Calculate melting point of a system."""
    while True:
        if task_output is None:
            task_output = await task_pmf_calculation(
                coordinate,
                temperature,
                flow_input,
                pmf_task_outputs,
                None,
                restart_time
            )

            task_output = await convergence_test_lindemann_index(
                flow_input,
                task_output,
                pmf_task_outputs,
                flow_input.flow_config.lindemann_n_last_frames
            )

        if task_output.convergence:
            return task_output
        else:
            task_output = None

        restart_time += 1


def melting_test_temperatures(t_min=None, t_max=None, t_step=100., n_temps=5):
    # with t_max included
    if t_max is not None:
        if t_min is None:
            t_min = t_max - (n_temps - 1) * t_step
        t_range = np.arange(t_min, t_max + t_step, t_step)
    else:
        if t_min is None:
            t_min = 400.
        t_range = np.arange(t_min, t_min + n_temps * t_step, t_step)
    return np.around(t_range, decimals=0)


async def subflow_melting_zone_search(
    coordinate: float,
    temperatures: List[float],
    flow_input: PMFInput,
    pmf_task_outputs: SafeList
) -> Tuple[List[int], List[float]]:
    """check and decide melting temperature zone.

    Args:
        coordinate (float): The coordinate controlled by the PMF calculation.
        temperatures (List[float]): Temperatures perfroming PMF calculation on.
        ldxs (List[float]): Lindemann index from melting point test.
        flow_input (PMFInput): PMF calculation configuration.
        step (_type_, optional): Temprature steps when setting new ones. Defaults to 100..
        n_temps (int, optional): Num of temperatures when extending zone. Defaults to 5.

    Returns:
        Tuple[List[int], List[float]]: _description_
    """
    # check if melting point is in the range
    t_step = flow_input.flow_config.t_step
    n_temps = flow_input.flow_config.n_temps
    _temperatures = np.sort(temperatures)
    await subflow_melting_coordinate(
        coordinate, temperatures, flow_input, pmf_task_outputs)
    new_task_outputs = await pmf_task_outputs.get_items()
    _ldxs = np.asarray([i.lindemann_index
                        for t in _temperatures for i in new_task_outputs
                        if(i.temperature == t) & (i.coordinate == coordinate)],
                       dtype=np.float32)

    melting_idx = np.where((_ldxs > 0.1) & (_ldxs < 0.3))[0]
    solid_idx = np.where(_ldxs <= 0.1)[0]
    liquid_idx = np.where(_ldxs >= 0.3)[0]
    if len(melting_idx) == 0:
        # if melting point is not in the range, extend the range
        if _ldxs.max() <= 0.1:
            old_temp_range = _temperatures[-2:]
            new_temp_range = melting_test_temperatures(
                t_min=_temperatures[-1], t_step=t_step, n_temps=n_temps
            )
            _temperatures = np.concatenate((old_temp_range, new_temp_range))
        elif _ldxs.min() >= 0.3:
            old_temp_range = _temperatures[:2]
            new_temp_range = melting_test_temperatures(
                t_max=_temperatures[0], t_step=t_step, n_temps=n_temps
            )
            _temperatures = np.concatenate((old_temp_range, new_temp_range))
        else:
            # if melting point is not in the range,
            # but both solid and liquid states are in the range,
            # extend the range to the melting point
            mid_step = np.abs(
                _temperatures[liquid_idx[0]] - _temperatures[solid_idx[-1]]
            )
            if (len(solid_idx) > 1) & (len(liquid_idx) > 1):
                solid_temp = _temperatures[solid_idx[-1]]
                old_temp_range = np.concatenate((
                    _temperatures[(solid_idx[-1] - 1):(solid_idx[-1] + 1)],
                    _temperatures[liquid_idx[0]:(liquid_idx[0] + 2)]
                ))
                new_temp_range = melting_test_temperatures(
                    t_min=solid_temp, t_step=mid_step/3, n_temps=3
                )
                _temperatures = np.concatenate(
                    (old_temp_range, new_temp_range))
                _temperatures = np.unique(_temperatures)
            elif len(liquid_idx) <= 1:
                solid_temp = _temperatures[solid_idx[-1]]
                old_temp_range = _temperatures[
                    (solid_idx[-1] - 1):(solid_idx[-1] + 1)
                ]
                liquid_extra = np.array([
                    _temperatures[liquid_idx[0]],
                    _temperatures[liquid_idx[0]] + t_step
                ])
                new_temp_range = melting_test_temperatures(
                    t_min=solid_temp, t_step=mid_step/3, n_temps=3
                )
                if len(solid_idx) <= 1:
                    solid_extra = np.array([
                        _temperatures[solid_idx[-1]] - t_step,
                        _temperatures[solid_idx[-1]]
                    ])
                    _temperatures = np.concatenate((
                        old_temp_range,
                        new_temp_range,
                        liquid_extra,
                        solid_extra
                    ))
                else:
                    _temperatures = np.concatenate((
                        old_temp_range,
                        new_temp_range,
                        liquid_extra
                    ))
                _temperatures = np.unique(_temperatures)
            else:
                liquid_temp = _temperatures[liquid_idx[0]]
                old_temp_range = _temperatures[
                    liquid_idx[0]:(liquid_idx[0] + 2)
                ]
                solid_extra = np.array([
                    _temperatures[solid_idx[-1]] - t_step,
                    _temperatures[solid_idx[-1]]
                ])
                new_temp_range = melting_test_temperatures(
                    t_max=liquid_temp, t_step=mid_step/3, n_temps=3
                )
                _temperatures = np.concatenate((
                    solid_extra,
                    new_temp_range,
                    old_temp_range
                ))
                _temperatures = np.unique(_temperatures)
        # recursively call the function
        new_temperatures = _temperatures.tolist()
        return await subflow_melting_zone_search(
            coordinate,
            new_temperatures,
            flow_input,
            pmf_task_outputs
        )
    else:
        # if melting point is in the range, return the melting point
        if (len(solid_idx) > 1) & (len(liquid_idx) > 1):
            return list(melting_idx), list(_temperatures)
        elif len(solid_idx) <= 1:
            if len(liquid_idx) != 0:
                old_temp_range = _temperatures[melting_idx[0]:liquid_idx[0]+2]
            else:
                old_temp_range = _temperatures[melting_idx[0]:]
            if len(solid_idx) != 0:
                solid_extra = np.array([
                    _temperatures[solid_idx[-1]],
                    _temperatures[solid_idx[-1]] - t_step
                ])
            else:
                solid_extra = np.array([
                    _temperatures[melting_idx[0]] - t_step * 2,
                    _temperatures[melting_idx[0]] - t_step
                ])
            _temperatures = np.concatenate((old_temp_range, solid_extra))
            _temperatures = np.unique(_temperatures)
        else:
            if len(solid_idx) != 0:
                old_temp_range = _temperatures[solid_idx[-2]:melting_idx[-1]+1]
            else:
                old_temp_range = _temperatures[:melting_idx[-1]+1]
            if len(liquid_idx) != 0:
                liquid_extra = np.array([
                    _temperatures[liquid_idx[0]],
                    _temperatures[liquid_idx[0]] + t_step
                ])
            else:
                liquid_extra = np.array([
                    _temperatures[melting_idx[-1]] + t_step,
                    _temperatures[melting_idx[-1]] + t_step * 2
                ])
            _temperatures = np.concatenate((liquid_extra, old_temp_range))
            _temperatures = np.unique(_temperatures)
        # recursively call the function
        new_temperatures = _temperatures.tolist()
        return await subflow_melting_zone_search(
            coordinate,
            new_temperatures,
            flow_input,
            pmf_task_outputs
        )


async def subflow_pmf_temperature_range(
    flow_input: PMFInput,
    pmf_task_outputs: SafeList
) -> List[float]:
    """Calculate the PMF of a given coordinate."""

    # parse IS, FS, TS coordinates
    is_coordinate = flow_input.flow_config.is_coordinate
    fs_coordinate = flow_input.flow_config.fs_coordinate
    ts_coordinate = flow_input.flow_config.ts_coordinate

    # read original temperature range
    if flow_input.flow_config.temperatures is not None:
        temperatures = np.array(flow_input.flow_config.temperatures)
    elif (flow_input.flow_config.t_min is not None) and (flow_input.flow_config.t_max is not None):
        temperatures = np.arange(
            flow_input.flow_config.t_min, flow_input.flow_config.t_max +
            flow_input.flow_config.t_step, flow_input.flow_config.t_step
        )
    elif (flow_input.flow_config.t_min is not None) and (flow_input.flow_config.t_max is None):
        temperatures = np.arange(
            flow_input.flow_config.t_min,
            flow_input.flow_config.t_min + flow_input.flow_config.n_temps *
            flow_input.flow_config.t_step,
            flow_input.flow_config.t_step
        )
    elif (flow_input.flow_config.t_min is None) and (flow_input.flow_config.t_max is not None):
        temperatures = np.arange(
            flow_input.flow_config.t_max -
            (flow_input.flow_config.n_temps - 1) *
            flow_input.flow_config.t_step,
            flow_input.flow_config.t_max + flow_input.flow_config.t_step,
            flow_input.flow_config.t_step
        )
    else:
        raise ValueError(
            'No temperature range is specified. '
            'Please specify either `temperatures` or `t_min` and `t_max`.'
        )

    if is_coordinate is None:
        is_coordinate = min(flow_input.flow_config.coordinates)
    if fs_coordinate is None:
        fs_coordinate = max(flow_input.flow_config.coordinates)

    # calculate melting points
    tasks = []
    # IS
    tasks.append(
        asyncio.ensure_future(subflow_melting_zone_search(
            is_coordinate, list(temperatures), flow_input, pmf_task_outputs
        ))
    )

    # FS
    tasks.append(
        asyncio.ensure_future(subflow_melting_zone_search(
            fs_coordinate, list(temperatures), flow_input, pmf_task_outputs
        ))
    )

    # TS
    if ts_coordinate is not None:
        tasks.append(
            asyncio.ensure_future(subflow_melting_zone_search(
                ts_coordinate, list(temperatures), flow_input, pmf_task_outputs
            ))
        )
    # wait for all tasks to finish
    await asyncio.gather(*tasks)

    # merging the melting points
    temps = [task.result()[1] for task in tasks]
    pmf_temperatures = np.concatenate(temps)
    pmf_temperatures = np.unique(pmf_temperatures)
    temperature_range = list(pmf_temperatures)

    return temperature_range


async def subflow_pmf_each_temperauture(
    temperature: float,
    flow_input: PMFInput,
    pmf_task_outputs: SafeList
):
    """define workflow calculating a single PMF at a given temperature"""

    for cursor, coordinate in enumerate(flow_input.flow_config.coordinates):
        # check if the PMF calculation is already done
        original_pmf_tasks = await pmf_task_outputs.get_items() # static list
        task_output = None
        init_structure_path = None
        task_check = get_item_from_list(
            original_pmf_tasks,
            coordinate=coordinate,
            temperature=temperature
        )
        if task_check is not None:
            task_output = task_check[1]
            if task_output.convergence:
                if task_output.pmf_mean is not None:
                    continue
        # check if the previous PMF calculation is done
        # if so, use the last frame as the initial structure
        if cursor != 0:
            last_task_check = get_item_from_list(
                original_pmf_tasks,
                coordinate=flow_input.flow_config.coordinates[cursor - 1],
                temperature=temperature
            )
            if last_task_check is not None:
                last_task_output = last_task_check[1]
                init_structure_path = last_task_output.last_frame_path

        restart_time = 0
        # submit the PMF calculation task
        while True:
            # if not, submit the PMF calculation task
            if task_output is not None:
                if task_output.convergence:
                    break
                else:
                    # not converged yet
                    # continue from current restart time
                    if task_output.restart_time >= restart_time:
                        restart_time = task_output.restart_time
                    task_output = await task_pmf_calculation(
                        coordinate,
                        temperature,
                        flow_input,
                        pmf_task_outputs,
                        init_structure_path,
                        restart_time
                    )
            else:
                # submit the PMF calculation task
                task_output = await task_pmf_calculation(
                    coordinate,
                    temperature,
                    flow_input,
                    pmf_task_outputs,
                    init_structure_path,
                    restart_time
                )

            # check if the PMF calculation is converged
            task_output = await convergence_test_lagrange_multiplier(
                task_output,  # type: ignore
                pmf_task_outputs
            )
            restart_time += 1


def plot_pmf_profile(
    flow_input: PMFInput,
    flow_output: PMFOutput
) -> None:
    """Plot the PMF of each state at each temperature."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    task_outputs = flow_output.pmf_tasks

    # plot the PMF of each state at each temperature
    for temperature in flow_output.temperature_range:
        pmf_means = np.zeros(len(flow_input.flow_config.coordinates))
        pmf_vars = np.zeros(len(flow_input.flow_config.coordinates))
        coordinates = np.array(flow_input.flow_config.coordinates)
        for i, coordinate in enumerate(coordinates):
            task_check = get_item_from_list(
                task_outputs, coordinate=coordinate, temperature=temperature
            )
            if task_check is not None:
                task_output = task_check[1]
                pmf_means[i] = task_output.pmf_mean
                pmf_vars[i] = task_output.pmf_var
        ax.errorbar(coordinates, pmf_means, pmf_vars, label=f'{temperature} K')
    ax.legend()
    ax.set_xlabel('Reaction Coordinate(Angstrom)')
    ax.set_ylabel('PMF (eV/Angstrom)')

    # save the figure
    work_path = Path(flow_input.job_config.work_path).resolve()
    plt.savefig(work_path / 'pmf.png')
