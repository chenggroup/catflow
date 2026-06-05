"""Test Pydantic configuration models."""

import os
from catflow.tasker.resources.config import (
    MachineConfig,
    SbatchResources,
    JobConfig,
    load_machine_config,
)


class TestSbatchResources:
    def test_default_resources(self):
        """Default SbatchResources has sensible defaults."""
        r = SbatchResources()
        assert r.partition == "cpu"
        assert r.node_count == 1
        assert r.cpu_per_node == 1
        assert r.gpu_per_node == 0
        assert r.custom_options == []
        assert r.envs == {}

    def test_render_slurm_header(self):
        """SLURM headers are correctly formatted."""
        r = SbatchResources(
            partition="gpu", node_count=2, cpu_per_node=8, gpu_per_node=2
        )
        headers = r.render_header("slurm")
        assert "#SBATCH --partition=gpu" in headers
        assert "#SBATCH --nodes=2" in headers
        assert "#SBATCH --ntasks-per-node=8" in headers
        assert "#SBATCH --gpus-per-node=2" in headers

    def test_custom_options_rendered(self):
        """Custom SBATCH options appear in headers."""
        r = SbatchResources(custom_options=["--exclusive", "--time=01:00:00"])
        headers = r.render_header("slurm")
        assert any("--exclusive" in h for h in headers)
        assert any("--time=01:00:00" in h for h in headers)

    def test_envs_rendered(self):
        """Environment variables are exported."""
        r = SbatchResources(envs={"OMP_NUM_THREADS": "4", "CUDA_VISIBLE_DEVICES": "0"})
        headers = r.render_header("slurm")
        assert any("OMP_NUM_THREADS=4" in h for h in headers)


class TestMachineConfig:
    def test_minimal_config(self):
        """Minimal MachineConfig with only name."""
        m = MachineConfig(name="local")
        assert m.name == "local"
        assert m.scheduler == "slurm"
        assert m.remote_executor is None

    def test_with_resources(self):
        """MachineConfig with custom resources."""
        m = MachineConfig(
            name="gpu-cluster",
            scheduler="slurm",
            resources=SbatchResources(partition="gpu", gpu_per_node=4),
        )
        assert m.resources.gpu_per_node == 4

    def test_load_machine_config(self, tmp_path):
        """YAML machine config is parsed correctly."""
        yaml_file = tmp_path / "machine.yml"
        yaml_file.write_text("""
machine:
  name: my-cluster
  scheduler: slurm
  resources:
    partition: gpu
    node_count: 2
""")
        m = load_machine_config(str(yaml_file))
        assert m.name == "my-cluster"
        assert m.resources.partition == "gpu"
        assert m.resources.node_count == 2

    def test_load_legacy_pool_format(self, tmp_path):
        """Legacy dynaconf POOL format is parsed."""
        yaml_file = tmp_path / "machine.yml"
        yaml_file.write_text("""
POOL:
  main:
    machine:
      queue_system: slurm
    resources:
      partition: gpu
      node_count: 1
""")
        m = load_machine_config(str(yaml_file))
        assert m.scheduler == "slurm"
        assert m.resources.partition == "gpu"


class TestJobConfig:
    def test_minimal_job_config(self):
        """Minimal JobConfig with only command."""
        j = JobConfig(command="echo hello")
        assert j.command == "echo hello"
        assert j.forward_files == []
        assert j.task_per_group == 1

    def test_job_config_with_files(self):
        """JobConfig with file lists."""
        j = JobConfig(
            command="lmp -i input.in",
            forward_files=["input.in", "conf.lmp"],
            backward_files=["log.lammps", "dump.lammpstrj"],
            task_per_group=4,
        )
        assert len(j.forward_files) == 2
        assert j.task_per_group == 4
