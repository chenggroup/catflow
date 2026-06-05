import json
import os
import warnings
import numpy as np

from collections import deque, defaultdict
from pathlib import Path
from ruamel.yaml import YAML
from typing import Union, List, Optional, Dict, Iterable

from ase import Atoms
from ase.io import read, write
from dpdata.unit import LengthConversion
from pymatgen.core.periodic_table import Element
from pydantic import BaseModel

from catflow.utils import logger
from catflow.tasker.resources.submit import JobFactory
from catflow.tasker.resources.config import MachineConfig, SbatchResources, JobConfig
from catflow.tasker.resources.script_gen import PmfBatchScript
from catflow.tasker.resources.workflow_executor import WorkflowExecutor
from catflow.utils.file import tail
from catflow.utils.cp2k import Cp2kInput, Cp2kInputToDict


class PMFTask(object):
    """
    Workflow for potential of mean force calculation.
    """

    def __init__(
            self,
            coordinate,
            temperature,
            init_structure_path: str,
            reaction_pair: Union[list, tuple],
            work_path: str,
            machine_name: str,
            resources: dict,
            command: str,
            input_dict=None,
            restart_time: int = 0,
            **kwargs
    ):
        self.coordinate = coordinate
        self.temperature = temperature
        self.reaction_pair = reaction_pair
        self.init_structure_path = Path(init_structure_path).resolve()
        self.init_structure = read(self.init_structure_path)
        self.work_path = Path(work_path).resolve()
        if not self.work_path.exists():
            os.makedirs(self.work_path)
        self.machine_name = machine_name
        self.resource_dict = resources
        self.command = command
        if kwargs:
            self.kwargs = kwargs
        else:
            self.kwargs = {}
        if input_dict is None:
            from .defaults.pmf import _default_input_dict
            input_dict = _default_input_dict
        self.input_dict = input_dict
        self.restart_time = restart_time
        if self.restart_time > 0:
            self.task_name = f'restart.{restart_time - 1}.{coordinate}_{temperature}'
        elif self.restart_time == 0:
            self.task_name = f'task.{coordinate}_{temperature}'
        else:
            raise ValueError(
                'Restart time should be an integer greater than 0.')
        self.task_path = self.work_path / self.task_name

    @property
    def cell(self):
        init_structure = self.init_structure
        if isinstance(init_structure, list):
            raise TypeError("Initial structure should be an Atoms instance")
        # define cell with x, y, z and alpha, beta, gamma
        if self.kwargs.get('cell') is None:
            cell = init_structure.cell.cellpar()
        else:
            cell = self.kwargs.get('cell')
            cell = np.array(cell)
        if cell.shape == (6,):
            pass
        elif cell.shape == (3, 3):
            from ase.geometry import Cell
            cell = Cell(cell).cellpar()
        elif cell.shape == (3,):
            # if cell is defined with x, y, z
            # set alpha, beta, gamma to 90
            cell = np.append(cell, [90, 90, 90])
        else:
            raise ValueError(
                'Cell should be defined with x, y, z (and alpha, beta, gamma).')
        return cell

    def generate_submission(self):
        # pass necessary parameters to task
        structure = self.init_structure
        coordinate = self.coordinate
        temperature = self.temperature

        if self.restart_time > 0:
            self._task_restart()
        else:
            self._task_preprocess()
            self._task_generate()
        self._task_postprocess()
        task_dict = self._get_task_generated(self.task_name)
        job = self.job_generator([task_dict])
        return job

    def get_last_frame(self):
        num_atoms = len(self.init_structure)
        logger.info('Reading end of file')
        with open(self.task_path / 'pmf-pos-1.xyz', 'rb') as f:
            last = tail(f, num_atoms + 2)
        last_path = self.task_path / 'last.xyz'
        logger.info(f'Writing last frame to {last_path}')
        with open(last_path, 'wb') as output:
            output.write(last)
        return last_path

    def _task_preprocess(self):
        pass

    def _task_generate(self):
        structure = self.init_structure
        coordinate = self.coordinate
        temperature = self.temperature

        os.makedirs(self.task_path, exist_ok=True)
        write(self.task_path / 'init.xyz', structure)
        logger.info(f"Writing init.xyz to {self.task_path}")

        input_dict = self._generate_input(coordinate, temperature)

        with open(self.task_path / 'input.inp', 'w') as f:
            output = Cp2kInput(params=input_dict).render()
            f.write(output)
        logger.info(f"Writing input.inp to {self.task_path}")

    def _generate_input(self, coordinate, temperature):
        conv = LengthConversion("angstrom", "bohr")
        coord = coordinate * conv.value()

        input_dict = self.input_dict

        # set job name
        input_dict["GLOBAL"]["PROJECT"] = "pmf"

        # set motion
        input_dict["MOTION"]["CONSTRAINT"]["COLLECTIVE"]["TARGET"] = coord
        input_dict["MOTION"]["MD"]["TEMPERATURE"] = temperature
        input_dict["FORCE_EVAL"]["SUBSYS"]["COLVAR"]["DISTANCE"]["ATOMS"] = \
            ' '.join([str(i + 1) for i in self.reaction_pair])
        a_b_c = self.cell[:3]
        alpha_beta_gamma = self.cell[3:]
        input_dict["FORCE_EVAL"]["SUBSYS"]["CELL"] = {
            "ABC": " ".join([str(i) for i in a_b_c]),
            "ALPHA_BETA_GAMMA": " ".join([str(i) for i in alpha_beta_gamma])
        }
        input_dict["FORCE_EVAL"]["SUBSYS"]["TOPOLOGY"] = {
            "COORD_FILE_FORMAT": "XYZ",
            "COORD_FILE_NAME": "init.xyz"
        }
        steps = self.kwargs.get('steps')
        timestep = self.kwargs.get('timestep')
        if steps:
            input_dict["MOTION"]["MD"]["STEPS"] = steps
            input_dict["MOTION"]["CONSTRAINT"]["LAGRANGE_MULTIPLIERS"][
                "COMMON_ITERATION_LEVELS"] = steps
        if timestep:
            input_dict["MOTION"]["MD"]["TIMESTEP"] = timestep # unit: fs

        dump_freq = self.kwargs.get('dump_freq')
        if dump_freq:
            md_print = input_dict.setdefault("MOTION", {}).setdefault("PRINT", {})
            md_print.setdefault("TRAJECTORY", {}).setdefault("EACH", {}).setdefault("MD", dump_freq)
            md_print.setdefault("FORCES", {}).setdefault("EACH", {}).setdefault("MD", dump_freq)
            input_dict["MOTION"]["PRINT"]["TRAJECTORY"]["EACH"]["MD"] = dump_freq
            input_dict["MOTION"]["PRINT"]["FORCES"]["EACH"]["MD"] = dump_freq
        return input_dict

    def _task_restart(self):
        import shutil

        os.makedirs(self.task_path, exist_ok=True)
        logger.info("Copying files from previous task")
        restart_file_list = ['init.xyz', 'pmf-1.restart']
        if self.restart_time < 2:
            # if restart_time is 1, copy from original task
            last_task_path = \
                self.task_path.parent / f'task.{self.coordinate}_{self.temperature}'
        else:
            # if restart_time is larger than 1, copy from previous restart task
            last_task_path = \
                self.task_path.parent / f'restart.{self.restart_time - 2}.{self.coordinate}_{self.temperature}'
        for file_name in restart_file_list:
            shutil.copy(last_task_path / file_name,self.task_path / file_name)
        restart_file = self.task_path / 'pmf-1.restart'
        restart_file.rename(self.task_path / 'pmf-1.original.restart')

        _restart_input = Cp2kInputToDict(
            self.task_path.parent /
            f'task.{self.coordinate}_{self.temperature}/input.inp')
        input_dict = _restart_input.get_tree()

        input_dict["MOTION"]["MD"]["STEPS"] = self.kwargs.get(
            'restart_steps', 10000000)
        if input_dict.get("EXT_RESTART"):
            input_dict["EXT_RESTART"]["RESTART_FILE_NAME"] = \
                'pmf-1.original.restart'
        else:
            input_dict["EXT_RESTART"] = {
                "RESTART_FILE_NAME": 'pmf-1.original.restart'
            }
        with open(self.task_path / 'input.inp', 'w') as f:
            output = Cp2kInput(params=input_dict).render()
            f.write(output)

    def _get_task_generated(self, task_name):
        forward_files = self.kwargs.get('forward_files', [])
        if forward_files is None:
            forward_files = ['input.inp', 'init.xyz']
        else:
            forward_files += ['input.inp', 'init.xyz']
        if self.restart_time > 0:
            forward_files += ['pmf-1.original.restart']
        backward_files = self.kwargs.get('backward_files')
        if backward_files is None:
            backward_files = ['pmf-1.ener', 'pmf.LagrangeMultLog',
             'pmf-pos-1.xyz', 'pmf-frc-1.xyz', 'output',
             'pmf-1.restart']
        outlog = self.kwargs.get('outlog')
        if outlog is None:
            outlog = 'output'

        errlog = self.kwargs.get('errlog')
        if errlog is None:
            errlog = 'err.log'

        task_dict = {
            "command": self.command,
            "task_work_path": task_name,
            "forward_files": forward_files,
            "backward_files": backward_files,
            "outlog": outlog,
            "errlog": errlog,
        }
        logger.info(f"{task_name} generating")
        return task_dict

    def _task_postprocess(self):
        pass

    def job_generator(self, task_dict_list):
        forward_common_files = self.kwargs.get('forward_common_files')
        if forward_common_files is None:
            forward_common_files = []
        backward_common_files = self.kwargs.get('backward_common_files')
        if backward_common_files is None:
            backward_common_files = []
        submission_dict = {
            "work_base": str(self.work_path),
            "forward_common_files": forward_common_files,
            "backward_common_files": backward_common_files
        }
        job = JobFactory(task_dict_list, submission_dict,
                         self.machine_name, self.resource_dict)
        return job.submission

    def generate_script(self, script_path: Optional[str] = None,
                        concurrency: int = 1) -> str:
        """Generate a bash script using oh-my-batch for PMF calculation.

        Replaces `generate_submission()` for the new oh-my-batch backend.
        Produces a script that calls `omb combo` + `omb batch` + `omb job`.

        Args:
            script_path: Path to write the script. Auto-generated if None.
            concurrency: Number of parallel tasks per batch group.

        Returns:
            Path to the generated bash script.
        """
        if self.restart_time > 0:
            self._task_restart()
        else:
            self._task_preprocess()
            self._task_generate()
        self._task_postprocess()

        # Build machine config from the existing kwargs/resources
        machine = MachineConfig(
            name=self.machine_name,
            scheduler="slurm",
            resources=SbatchResources(
                partition=self.resource_dict.get("partition", "cpu"),
                node_count=self.resource_dict.get("node_count", 1),
                cpu_per_node=self.resource_dict.get("cpu_per_node", 1),
                gpu_per_node=self.resource_dict.get("gpu_per_node", 0),
            ),
        )

        script = PmfBatchScript(
            machine=machine,
            command=self.command,
            forward_files=self.kwargs.get("forward_files", []),
            backward_files=self.kwargs.get("backward_files", []),
        )

        if script_path is None:
            script_path = str(self.task_path / "run_pmf.sh")

        script.build_pmf_script(
            coordinates=[self.coordinate],
            temperatures=[self.temperature],
            template_dir=str(self.task_path),
            output_dir=str(self.task_path),
            concurrency=concurrency,
        )
        return script.write(script_path)

    def check_point(self):
        """Checkpoint the current state of the task.

        Uses ai2-kit's @apply_checkpoint mechanism for resumability.
        """
        from catflow.core import set_checkpoint_dir
        set_checkpoint_dir(str(self.work_path / ".ckpt"))
        pass

    def input_dict_from_file(self, template_file):
        _cp2k_input = Cp2kInputToDict(template_file)
        _input_dict = _cp2k_input.get_tree()
        self.input_dict = _input_dict


class DPPMFTask(PMFTask):
    """calculate the free energy of a reaction coordinate using DeePMD-kit
    """

    def type_map(self) -> dict:
        type_map = self.kwargs.get("type_map")
        if isinstance(type_map, dict):
            return type_map
        else:
            structure = self.init_structure
            if isinstance(structure, list):
                raise TypeError("Initial structure should be an Atoms instance")
            element_set = set(structure.symbols)
            type_map = {}
            for i, element in enumerate(element_set):
                type_map[element] = i
        return type_map

    def _link_model(self, model_path):
        model_abs_path = Path(model_path).resolve()
        graph_file = self.work_path / 'graph.pb'
        if not graph_file.exists():
            graph_file.symlink_to(model_abs_path)
        else:
            logger.info("model symlink exists")
        fwd_files_flag = self.kwargs.get("forward_common_files")
        if fwd_files_flag:
            self.kwargs["forward_common_files"] += ['graph.pb']
        else:
            self.kwargs["forward_common_files"] = ['graph.pb']

    def _make_charge_dict(self, structure):
        element_set = set(structure.symbols)
        self.input_dict["FORCE_EVAL"]["MM"]["FORCEFIELD"]["CHARGE"] = []
        for element in element_set:
            if Element(element).is_metal:
                _charge_dict = {
                    "ATOM": element,
                    "CHARGE": 0.0
                }
                self.input_dict["FORCE_EVAL"]["MM"]["FORCEFIELD"]["CHARGE"].append(
                    _charge_dict)

    def _make_deepmd_dict(self, structure):
        element_set = set(structure.symbols)
        self.input_dict["FORCE_EVAL"]["MM"]["FORCEFIELD"]["NONBONDED"][
            "DEEPMD"] = []
        for element in element_set:
            try:
                type_map = self.type_map()
                _deepmd_dict = {
                    "ATOMS": element + ' ' + element,
                    "POT_FILE_NAME": "../graph.pb",
                    "ATOM_DEEPMD_TYPE": type_map[element]
                }
            except KeyError:
                raise KeyError("Please provide type_map property")
            self.input_dict["FORCE_EVAL"]["MM"]["FORCEFIELD"]["NONBONDED"][
                "DEEPMD"].append(_deepmd_dict)

    def _task_preprocess(self):
        structure = self.init_structure
        self._make_charge_dict(structure)
        self._make_deepmd_dict(structure)
    
    def _task_postprocess(self):
        self._link_model(self.kwargs.get('model_path'))
