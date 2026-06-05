"""Machine and job configuration models for CatFlow.

Replaces the previous Dynaconf-based configuration with Pydantic models.
Supports both legacy YAML format (machine.yml) and new format.
"""

import sys
import warnings
from pathlib import Path
from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class SbatchResources(BaseModel):
    """SBATCH resource directives for job submission."""
    partition: str = "cpu"
    node_count: int = 1
    cpu_per_node: int = 1
    gpu_per_node: int = 0
    custom_options: List[str] = Field(default_factory=list)
    envs: Dict[str, str] = Field(default_factory=dict)

    def render_header(self, scheduler: str = "slurm") -> List[str]:
        """Render SBATCH headers as a list of directive strings."""
        if scheduler == "slurm":
            headers = [
                f"#SBATCH --partition={self.partition}",
                f"#SBATCH --nodes={self.node_count}",
                f"#SBATCH --ntasks-per-node={self.cpu_per_node}",
            ]
            if self.gpu_per_node > 0:
                headers.append(f"#SBATCH --gpus-per-node={self.gpu_per_node}")
            for opt in self.custom_options:
                headers.append(f"#SBATCH {opt}")
            for key, val in self.envs.items():
                headers.append(f"export {key}={val}")
            return headers
        elif scheduler == "lsf":
            # TODO: LSF header rendering
            return [f"#BSUB -n {self.cpu_per_node}", f"#BSUB -q {self.partition}"]
        elif scheduler == "openpbs":
            # TODO: PBS header rendering
            return [f"#PBS -l nodes={self.node_count}:ppn={self.cpu_per_node}"]
        return []


class MachineConfig(BaseModel):
    """Configuration for a remote/local machine and its scheduler."""
    name: str
    scheduler: str = "slurm"  # slurm, lsf, openpbs
    resources: SbatchResources = Field(default_factory=SbatchResources)
    remote_executor: Optional[Dict] = Field(
        default=None,
        description="ai2-kit executor config for SSH (host, user, key, etc.)"
    )


class JobConfig(BaseModel):
    """Configuration for a job submission task group."""
    command: str
    forward_files: List[str] = Field(default_factory=list)
    backward_files: List[str] = Field(default_factory=list)
    forward_common_files: List[str] = Field(default_factory=list)
    backward_common_files: List[str] = Field(default_factory=list)
    task_per_group: int = Field(
        default=1,
        description="Number of tasks per batch script group (omb batch --concurrency)"
    )


def load_machine_config(path: str) -> MachineConfig:
    """Load machine configuration from a YAML file.

    Supports both legacy (dynaconf-style) and new Pydantic-based format.
    """
    from ruamel.yaml import YAML
    yaml = YAML(typ='safe')
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Machine config not found: {path}")

    with open(path) as f:
        data = yaml.load(f)

    # New-style config: top-level 'machine' key
    if isinstance(data, dict) and 'machine' in data:
        return MachineConfig(**data['machine'])

    # Legacy dynaconf-style: key contains 'machine' sub-key with POOL
    # Try to extract from dynaconf pool format
    if isinstance(data, dict) and 'POOL' in data:
        pool = data['POOL']
        first_key = list(pool.keys())[0]
        entry = pool[first_key]
        machine_dict = entry.get('machine', {})
        resources_dict = entry.get('resources', {})
        return MachineConfig(
            name=first_key,
            scheduler=machine_dict.get('queue_system', 'slurm'),
            resources=SbatchResources(
                partition=resources_dict.get('partition', 'cpu'),
                node_count=resources_dict.get('node_count', 1),
                cpu_per_node=resources_dict.get('cpu_per_node', 1),
                gpu_per_node=resources_dict.get('gpu_per_node', 0),
            )
        )

    # Direct MachineConfig fields (name + scheduler + resources inline)
    if isinstance(data, dict) and 'name' in data:
        return MachineConfig(**data)

    raise ValueError(
        f"Cannot parse machine config from {path}. "
        "Expected format: machine.yml with 'machine' key, "
        "'POOL' key (legacy dynaconf), or direct MachineConfig fields."
    )


def find_machine_config(paths: List[str] = None) -> Optional[str]:
    """Search common paths for machine configuration."""
    if paths is None:
        paths = [
            "machine.yml",
            "machine.local.yml",
            str(Path(sys.prefix) / "etc" / "miko" / "machine.yml"),
        ]
    for p in paths:
        p_path = Path(p).resolve()
        if p_path.exists():
            return str(p_path)
    return None


# --- Backward compatibility shim ---
# The old `settings` object (Dynaconf) was replaced by Pydantic models.
# This provides a deprecation path for modules that still import `settings`.
def _get_settings():
    """Deprecated: use load_machine_config() instead."""
    warnings.warn(
        "'settings' from config is deprecated. Use load_machine_config() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return {}


class settings_shim:
    """Backward-compatible shim for old settings imports."""

    def __getitem__(self, key):
        return _get_settings()

    def __getattr__(self, name):
        return _get_settings()


settings = settings_shim()
