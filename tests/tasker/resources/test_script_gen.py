"""Test script generators for omb-based bash script generation."""

from pathlib import Path

from catflow.tasker.resources.script_gen import (
    PmfBatchScript,
    TeslaBatchScript,
    ScriptBuilder,
)
from catflow.tasker.resources.config import MachineConfig, SbatchResources


class TestScriptBuilder:
    def test_render_basic(self):
        """Simple script renders correctly."""
        sb = ScriptBuilder()
        # _header_lines and _body_lines initialized manually
        sb._header_lines = []
        sb._body_lines = []
        sb.add_body("echo hello")
        output = sb.render()
        assert output.startswith("#!/bin/bash")
        assert "set -e" in output
        assert "echo hello" in output

    def test_sbatch_header(self):
        """SBATCH directives are included when machine config provided."""
        machine = MachineConfig(
            name="test",
            scheduler="slurm",
            resources=SbatchResources(
                partition="gpu", node_count=2, cpu_per_node=4, gpu_per_node=1
            ),
        )
        sb = ScriptBuilder()
        sb._header_lines = []
        sb._body_lines = []
        sb.add_sbatch_header(machine)
        output = sb.render()

        assert "#SBATCH --partition=gpu" in output
        assert "#SBATCH --nodes=2" in output
        assert "#SBATCH --ntasks-per-node=4" in output
        assert "#SBATCH --gpus-per-node=1" in output

    def test_write_script(self, tmp_path):
        """write() creates an executable file."""
        sb = ScriptBuilder()
        sb._header_lines = []
        sb._body_lines = []
        sb.add_body("echo test")

        path = sb.write(str(tmp_path / "run.sh"))
        assert Path(path).exists()
        assert Path(path).stat().st_mode & 0o111  # executable


class TestPmfBatchScript:
    def test_build_pmf_script(self):
        """PmfBatchScript generates correct omb combo → batch → job pipeline."""
        machine = MachineConfig(
            name="test",
            scheduler="slurm",
            resources=SbatchResources(partition="gpu", node_count=1, cpu_per_node=4),
        )
        script = PmfBatchScript(machine=machine, command="cp2k.ssmp -i input.inp")
        output = script.build_pmf_script(
            coordinates=[1.4, 1.8, 2.2],
            temperatures=[300, 500],
            template_dir="./tpl",
            output_dir="./pmf_out",
            concurrency=4,
        )

        # Check omb combo section
        assert "omb combo" in output
        assert "COORDINATE 1.4 1.8 2.2" in output
        assert "TEMPERATURE 300 500" in output
        assert "make_files" in output

        # Check omb batch section
        assert "omb batch" in output
        assert "add_work_dirs" in output
        assert "cp2k.ssmp -i input.inp" in output
        assert "--concurrency 4" in output

        # Check omb job section
        assert "omb job slurm submit" in output
        assert "--max_tries 3" in output
        assert "--recovery pmf_recovery.json" in output

    def test_empty_coordinates(self):
        """Script generation handles empty lists gracefully."""
        machine = MachineConfig(name="test", scheduler="slurm")
        script = PmfBatchScript(machine=machine, command="echo")
        # Should not crash, just produce an omb combo with no vars
        output = script.build_pmf_script(
            coordinates=[], temperatures=[], template_dir="./tpl", output_dir="./x"
        )
        assert isinstance(output, str)
        assert "omb combo" in output


class TestTeslaBatchScript:
    def test_train_stage_script(self):
        """Train stage generates omb combo + batch + job for training."""
        machine = MachineConfig(
            name="test",
            scheduler="slurm",
            resources=SbatchResources(partition="gpu", cpu_per_node=4, gpu_per_node=1),
        )
        script = TeslaBatchScript(machine=machine)
        script.train_stage(
            work_dir="/tmp/test", model_count=4, train_steps=400000, concurrency=4
        )
        output = script.render()

        # Should reference model directories
        assert "omb combo" in output
        assert "SEED" in output
        assert "STEPS" in output
        assert "omb batch" in output
        assert "bash run.sh" in output
        assert "omb job slurm submit" in output
        assert "--max_tries 2" in output
        assert "--recovery" in output

    def test_explore_stage_script(self):
        """Explore stage generates correct batch + job for LAMMPS."""
        machine = MachineConfig(name="test", scheduler="slurm")
        script = TeslaBatchScript(machine=machine)
        script.explore_stage(
            work_dir="/tmp/test", iter_name="000", concurrency=5
        )
        output = script.render()

        assert "01.model_devi" in output or "EXPLORE_DIR" in output
        assert "omb batch" in output
        assert "add_cmd" in output
        assert "bash run.sh" in output
        assert "omb job slurm submit" in output
        assert "--max_tries 2" in output

    def test_labeling_stage_script(self):
        """Label stage generates correct batch + job for CP2K/VASP."""
        machine = MachineConfig(name="test", scheduler="slurm")
        script = TeslaBatchScript(machine=machine)
        script.labeling_stage(
            work_dir="/tmp/test", iter_name="000", software="cp2k", concurrency=5
        )
        output = script.render()

        assert "02.fp" in output or "LABEL_DIR" in output
        assert "cp2k.ssmp -i input.inp" in output
        assert "omb batch" in output
        assert "omb job slurm submit" in output
        assert "--max_tries 3" in output
