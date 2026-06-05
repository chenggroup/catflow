import os
import shutil

from copy import deepcopy
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from catflow.utils.log_factory import logger
from catflow.tasker.resources.submit import JobFactory
from catflow.tasker.resources.config import MachineConfig, SbatchResources, JobConfig
from catflow.tasker.resources.script_gen import ScriptBuilder
from catflow.tasker.resources.workflow_executor import WorkflowExecutor


def trajectory_checkrun_vasp(
        traj_file: str, 
        work_path: str, 
        chemical_symbol: Optional[str] = None, 
        cell: Optional[Union[List, np.ndarray]] = None,
        index: str = "::",
        **task_generation_params
    ):
    """Exports VASP POSCAR files from a trajectory.

    Args:
        traj_file (str): The trajectory file to read.
        work_path (str): The working directory to write the POSCAR files to.
        chemical_symbol (str): The chemical symbol to use for the atoms in the POSCAR files.
        index (str): The index of the frame to use.

    Returns:
        None
    """
    from ase.io import iread, write

    stcs = iread(os.path.abspath(traj_file), index=index)    
    os.makedirs(work_path, exist_ok=True)

    for i, j in enumerate(stcs):
        if chemical_symbol:
            j.set_chemical_symbols(chemical_symbol)
        if cell is not None:
            j.set_cell(cell)
        task_path = os.path.join(work_path, f'task.{str(i).zfill(6)}')
        os.makedirs(task_path, exist_ok=True)
        write(os.path.join(task_path, 'POSCAR'), j, vasp5=True) # type: ignore
    if task_generation_params:
        submission = multi_fp_task(work_path=work_path, **task_generation_params)
        return submission


def multi_fp_task(
        work_path, 
        fp_command, 
        machine_name, 
        resource_dict, 
        files_to_forward: Optional[List[str]] = None,
        **kwargs
    ):
    """Run multiple VASP tasks in parallel.

    Args:
        work_path (_type_): _description_
        fp_command (_type_): _description_
        machine_name (_type_): _description_
        resource_dict (_type_): _description_
    """
    forward_files = kwargs.get('forward_files', [])
    forward_files += ['POSCAR', 'INCAR', 'POTCAR']
    forward_files = list(set(forward_files))

    backward_files = kwargs.get('backward_files', [])
    backward_files += ['OUTCAR', 'vasprun.xml', 'fp.log', 'fp.err']
    backward_files = list(set(backward_files))

    task_dir_pattern = kwargs.get('task_dir_pattern', 'task.*')
    fp_tasks = glob(os.path.join(work_path, task_dir_pattern))
    fp_tasks.sort()
    if len(fp_tasks) == 0:
        return
    if files_to_forward is not None:
        for ii in fp_tasks:
            for jj in files_to_forward:
                shutil.copy(jj, ii)
    fp_run_tasks = fp_tasks
    run_tasks = [os.path.basename(ii) for ii in fp_run_tasks]

    task_dict_list = [
        {
            "command": fp_command,
            "task_work_path": task,
            "forward_files": forward_files,
            "backward_files": backward_files,
            "outlog": "fp.log",
            "errlog": "fp.err",
        } for task in run_tasks
    ]

    submission_dict = {
        "work_base": work_path,
        "forward_common_files": kwargs.get('forward_common_files', []),
        "backward_common_files": kwargs.get('backward_common_files', [])
    }
    job = JobFactory(task_dict_list, submission_dict, machine_name, resource_dict, group_size=1)
    submission = job.submission
    return submission.run_submission(exit_on_submit=True)


def cell_tests(
        init_structure_path,
        work_path,
        cell_list,
        cubic=True,
        required_files=None,
        **task_generation_params
):
    """Run VASP tasks with different cell sizes.
    """

    from ase.io import read, write

    work_path = Path(work_path).resolve()
    if required_files is None:
        required_files = {
            "incar_path": os.path.join(work_path, "INCAR"),
            "potcar_path": os.path.join(work_path, "POTCAR")
        }
    if cubic is False:
        for c in cell_list:
            if np.array(c).shape != (3, 3):
                raise Exception("should provide 3 vectors as cell")
    init_structure = read(os.path.abspath(init_structure_path))
    for idx, cell in enumerate(cell_list):
        task_path = os.path.join(os.path.abspath(work_path), f'task.{str(idx).zfill(3)}')
        os.makedirs(task_path, exist_ok=True)
        s = deepcopy(init_structure)
        s.set_cell(cell) # type: ignore
        s.set_pbc([1, 1, 1]) # type: ignore
        write(os.path.join(task_path, 'POSCAR'), s, vasp5=True) # type: ignore
        assert ('incar_path' in required_files.keys()) & ('potcar_path' in required_files.keys())
        for path in required_files.values():
            shutil.copy(
                path,
                os.path.join(task_path, os.path.basename(path))
            )
    multi_fp_task(work_path=work_path, **task_generation_params)


def multi_fp_task_omb(
    work_path,
    fp_command,
    machine_name="default",
    concurrency=4,
    task_pattern="task.*",
    forward_files=None,
    backward_files=None,
    **kwargs
):
    """Submit multiple VASP/CP2K FP tasks via oh-my-batch.

    Generates an `omb batch` + `omb job` script for the existing
    task directories under work_path.

    Args:
        work_path: Directory containing task.* subdirectories.
        fp_command: Command to run in each task directory.
        machine_name: Machine name (for logging).
        concurrency: Number of parallel tasks per batch script.
        task_pattern: Glob pattern for task directories.
        forward_files: Files to send to each task directory.
        backward_files: Files to retrieve from each task directory.
    """
    work_path = Path(work_path).resolve()
    task_dirs = sorted(glob(str(work_path / task_pattern)))

    if not task_dirs:
        logger.warning(f"No tasks found in {work_path}/{task_pattern}")
        return

    logger.info(f"[OMB] Found {len(task_dirs)} tasks in {work_path}")

    # Generate a batch script using omb
    sb = ScriptBuilder()

    defaults = kwargs.get('resources', {})
    sb.add_sbatch_header(MachineConfig(
        name=machine_name,
        scheduler=kwargs.get('scheduler', 'slurm'),
        resources=SbatchResources(
            partition=defaults.get('partition', 'cpu'),
            node_count=defaults.get('node_count', 1),
            cpu_per_node=defaults.get('cpu_per_node', 8),
            gpu_per_node=defaults.get('gpu_per_node', 0),
        ),
    ))

    sb.add_comment("FP task submission via oh-my-batch")
    sb.add_body(f'FP_WORK_DIR="{work_path}"')

    # omb batch: pack tasks
    batch_cmd = f'omb batch add_work_dirs "{work_path}/{task_pattern}"'
    if kwargs.get('header_file'):
        batch_cmd += f' add_header_files "{kwargs["header_file"]}"'
    batch_cmd += f' add_cmd "{fp_command}"'
    batch_cmd += f' make "{work_path}/fp-{{i}}.slurm" --concurrency {concurrency}'
    sb.add_body(batch_cmd)

    # omb job: submit with recovery
    recovery = kwargs.get('recovery_file', f'{work_path.name}_recovery.json')
    sb.add_body(
        f'omb job slurm submit "{work_path}/fp-*.slurm" '
        f'--max_tries 3 --wait --recovery "{work_path}/{recovery}"'
    )

    script_path = sb.write(str(work_path / "run_omb_fp.sh"))
    logger.info(f"[OMB] Script written: {script_path}")

    executor = WorkflowExecutor.local(work_base=str(work_path))
    result = executor.run_script(script_path)
    if result.returncode != 0:
        raise RuntimeError(f"OMB FP task failed: {result.stderr}")
    logger.info("[OMB] FP tasks completed successfully.")
