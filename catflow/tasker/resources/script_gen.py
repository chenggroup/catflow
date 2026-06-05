"""Script generators for CatFlow workflows.

Generates bash scripts that call oh-my-batch (omb) commands for
parameter combination, batch script generation, and job submission.
"""

import os
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field

from catflow.tasker.resources.config import MachineConfig, JobConfig


def _render_cmd(command: str, cwd: Optional[str] = None) -> str:
    """Render a command with optional cwd."""
    if cwd:
        return f"cd {cwd} && {command}"
    return command


class ScriptBuilder:
    """Base class for script generation with common utilities.

    Not a dataclass itself to avoid inheritance issues with dataclass fields.
    Subclasses are dataclasses that inherit from this mixin.
    """

    def __post_init_script(self):
        """Initialize script builder state (call from __post_init__)."""
        if not hasattr(self, '_header_lines'):
            self._header_lines = []
        if not hasattr(self, '_body_lines'):
            self._body_lines = []

    def add_header(self, line: str):
        self._header_lines.append(line)

    def add_body(self, line: str):
        self._body_lines.append(line)

    def add_comment(self, comment: str):
        self._body_lines.append(f"# {comment}")

    def add_empty(self):
        self._body_lines.append("")

    def add_sbatch_header(self, machine):
        """Add SBATCH directives from machine config."""
        for h in machine.resources.render_header(machine.scheduler):
            self.add_header(h)

    def add_module_loads(self, modules: List[str]):
        """Add module load commands."""
        for m in modules:
            self.add_body(f"module load {m}")

    def add_env_setup(self, envs: dict):
        """Add environment variable exports."""
        for k, v in envs.items():
            self.add_body(f"export {k}={v}")

    def render(self) -> str:
        """Render the complete bash script."""
        lines = ["#!/bin/bash", "set -e"]
        if self._header_lines:
            lines.extend(self._header_lines)
            lines.append("")
        lines.extend(self._body_lines)
        return "\n".join(lines) + "\n"

    def write(self, path: str) -> str:
        """Write script to file and return the path."""
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write(self.render())
        os.chmod(path, 0o755)
        return str(path)


@dataclass
class PmfBatchScript(ScriptBuilder):
    """Generate bash scripts for PMF (Potential of Mean Force) workflow.

    Produces scripts that use `omb combo`, `omb batch`, and `omb job`
    to run constrained MD simulations at specified coordinates and temperatures.
    """

    machine: MachineConfig
    command: str
    forward_files: List[str] = field(default_factory=list)
    backward_files: List[str] = field(default_factory=list)

    def __post_init__(self):
        self._header_lines = []
        self._body_lines = []
        self.add_sbatch_header(self.machine)

    def setup_combo(self, coordinates: List[float], temperatures: List[float],
                    template_dir: str, output_dir: str):
        """Generate omb combo section for coordinate × temperature grid."""
        self.add_comment("Generate task directories with omb combo")
        coords_str = " ".join(str(c) for c in coordinates)
        temps_str = " ".join(str(t) for t in temperatures)

        combo_cmd = (
            f"omb combo \\\n"
            f"  add_var COORDINATE {coords_str} \\\n"
            f"  add_var TEMPERATURE {temps_str} \\\n"
            f"  add_seq SEED --start 1 --stop 5 --step 1 \\\n"
            f"  set_broadcast SEED \\\n"
            f"  make_files {output_dir}/task.{{i}}-COORD-{{COORDINATE}}-TEMP-{{TEMPERATURE}}/input.inp "
            f"--template {template_dir}/input.inp.template \\\n"
            f"  done"
        )
        self.add_body(combo_cmd)
        self.add_empty()

    def setup_batch(self, work_dir: str, concurrency: int = 4,
                    header_file: Optional[str] = None):
        """Generate omb batch section to pack tasks into batch scripts."""
        self.add_comment("Pack tasks into batch scripts with omb batch")
        batch_cmd = (
            f"omb batch \\\n"
            f"  add_work_dirs {work_dir}/task.* \\\n"
        )
        if header_file:
            batch_cmd += f"  add_header_files {header_file} \\\n"
        else:
            # Render inline headers
            for h in self._header_lines:
                batch_cmd += f"  add_headers '{h}' \\\n"
        batch_cmd += (
            f"  add_cmd '{self.command}' \\\n"
            f"  make {work_dir}/batch-{{i}}.slurm --concurrency {concurrency}"
        )
        self.add_body(batch_cmd)
        self.add_empty()

    def submit_jobs(self, work_dir: str, max_tries: int = 3,
                    recovery_file: str = "pmf_recovery.json"):
        """Generate omb job submit section."""
        self.add_comment("Submit batch jobs via omb job")
        submit_cmd = (
            f"omb job slurm submit {work_dir}/batch-*.slurm \\\n"
            f"  --max_tries {max_tries} --wait --recovery {recovery_file}"
        )
        self.add_body(submit_cmd)
        self.add_empty()

    def build_pmf_script(self, coordinates: List[float], temperatures: List[float],
                         template_dir: str, output_dir: str,
                         concurrency: int = 4, max_tries: int = 3) -> str:
        """Build a complete PMF computation script."""
        self.setup_combo(coordinates, temperatures, template_dir, output_dir)
        self.setup_batch(output_dir, concurrency)
        self.submit_jobs(output_dir, max_tries)
        return self.render()


@dataclass
class TeslaBatchScript(ScriptBuilder):
    """Generate bash scripts for TESLA workflow stages.

    Produces scripts for training (DeepMD), exploration (LAMMPS),
    and labeling (CP2K/VASP) stages using oh-my-batch.
    """

    machine: MachineConfig

    def __post_init__(self):
        self._header_lines = []
        self._body_lines = []
        self.add_sbatch_header(self.machine)

    def train_stage(self, work_dir: str, model_count: int = 4,
                    train_steps: int = 400000, concurrency: int = 4):
        """Generate training stage script (DeepMD).

        Assumes model directories (000/, 001/, ...) with input.json
        have already been created by Python _make_template or dpgen.
        This generates run.sh, packs into batch scripts, and submits.
        """
        self.add_comment("=== Training Stage ===")
        self.add_body(f"TRAIN_DIR={work_dir}/00.train")
        self.add_empty()

        # Create run.sh in each model directory (seed substitution via combo)
        self.add_comment("Generate run.sh for each model")
        self.add_body(
            f"omb combo \\\n"
            f"  add_randint SEED -n {model_count} -a 0 -b 999999 \\\n"
            f"  add_var STEPS {train_steps} \\\n"
            f"  set_broadcast STEPS \\\n"
            f"  make_files $TRAIN_DIR/{{i}}/run.sh "
            f"--template templates/deepmd/run.sh --mode 755 \\\n"
            f"  done"
        )
        self.add_empty()

        # Pack into batch scripts
        self.add_comment("Pack training tasks")
        self.add_body(
            f"omb batch \\\n"
            f"  add_work_dirs $TRAIN_DIR/000 $TRAIN_DIR/001 $TRAIN_DIR/002 $TRAIN_DIR/003 \\\n"
            f"  add_cmd 'bash run.sh' \\\n"
            f"  make $TRAIN_DIR/train-{{i}}.slurm --concurrency {concurrency}"
        )
        self.add_empty()

        # Submit with recovery
        self.add_comment("Submit training jobs")
        self.add_body(
            f"omb job slurm submit $TRAIN_DIR/train-*.slurm \\\n"
            f"  --max_tries 2 --wait --recovery $TRAIN_DIR/.recovery.json"
        )

    def explore_stage(self, work_dir: str, iter_name: str,
                      concurrency: int = 5):
        """Generate exploration stage script (LAMMPS).

        Assumes task.* directories with input.lammps and conf.lmp
        have already been created by Python _make_template or dpgen.
        This generates run.sh, packs into batch scripts, and submits.
        """
        self.add_comment("=== Exploration Stage ===")
        iter_dir = f"{work_dir}/iter-{iter_name}"
        exp_dir = f"{iter_dir}/01.model_devi"
        self.add_body(f"EXPLORE_DIR={exp_dir}")
        self.add_empty()

        # Create run.sh in each task directory
        self.add_comment("Create run.sh in each exploration task")
        self.add_body(
            f"for task_dir in $EXPLORE_DIR/task.*; do\n"
            f"  cp templates/lammps/run.sh $task_dir/run.sh\n"
            f"  chmod 755 $task_dir/run.sh\n"
            f"  echo '[OMB] Created run.sh in $task_dir'\n"
            f"done"
        )
        self.add_empty()

        # Pack into batch scripts
        self.add_comment("Pack exploration tasks")
        self.add_body(
            f"omb batch \\\n"
            f"  add_work_dirs $EXPLORE_DIR/task.* \\\n"
            f"  add_cmd 'bash run.sh' \\\n"
            f"  make $EXPLORE_DIR/explore-{{i}}.slurm --concurrency {concurrency}"
        )
        self.add_empty()

        # Submit
        self.add_comment("Submit exploration jobs")
        self.add_body(
            f"omb job slurm submit $EXPLORE_DIR/explore-*.slurm \\\n"
            f"  --max_tries 2 --wait --recovery $EXPLORE_DIR/.recovery.json"
        )

    def labeling_stage(self, work_dir: str, iter_name: str,
                       software: str = "cp2k", concurrency: int = 5):
        """Generate labeling stage script (CP2K or VASP).

        Assumes task.* directories with POSCAR/coord.xyz and input files
        have already been created by Python _make_template or dpgen.
        """
        self.add_comment(f"=== Labeling Stage ({software}) ===")
        iter_dir = f"{work_dir}/iter-{iter_name}"
        label_dir = f"{iter_dir}/02.fp"
        self.add_body(f"LABEL_DIR={label_dir}")
        self.add_empty()

        # Detect and use appropriate command
        self.add_comment("Detect FP software")
        if software == "cp2k":
            run_cmd = "cp2k.ssmp -i input.inp"
        else:
            run_cmd = "mpirun vasp_std"

        # Pack and submit
        self.add_body(
            f"omb batch \\\n"
            f"  add_work_dirs $LABEL_DIR/task.* \\\n"
            f"  add_cmd '{run_cmd}' \\\n"
            f"  make $LABEL_DIR/label-{{i}}.slurm --concurrency {concurrency}"
        )
        self.add_empty()

        self.add_comment("Submit labeling jobs")
        self.add_body(
            f"omb job slurm submit $LABEL_DIR/label-*.slurm \\\n"
            f"  --max_tries 3 --wait --recovery $LABEL_DIR/.recovery.json"
        )

    def data_collection(self, work_dir: str, iter_name: str,
                        type_map: str = "[Ag,O]"):
        """Generate data collection step using ai2-kit tools."""
        self.add_comment("=== Data Collection ===")
        iter_dir = f"{work_dir}/iter-{iter_name}"
        self.add_body(f"ITER_DIR={iter_dir}")
        self.add_empty()

        self.add_comment("Convert labeling output to dpdata")
        self.add_body(
            f"ai2-kit tool dpdata read $ITER_DIR/02.label/job-*/output \\\n"
            f"  --fmt='cp2k/output' --type_map '{type_map}' \\\n"
            f"  write $ITER_DIR/03.data --fmt='deepmd/npy'"
        )
        self.add_empty()

        self.add_comment("Mark iteration complete")
        self.add_body(f"touch $ITER_DIR/iter.done")
