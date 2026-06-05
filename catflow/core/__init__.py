"""Core abstractions for CatFlow, adapted from ai2-kit.

Re-exports and thin wrappers around ai2-kit's core primitives:
- CheckpointService / @apply_checkpoint: function-level checkpointing for resumable workflows
- Artifact: data reference with URL, format, and metadata
- BashScript / BashStep: bash script generation with SBATCH header support
"""

from ai2_kit.core.checkpoint import apply_checkpoint, checkpoint_service
from ai2_kit.core.artifact import Artifact
from ai2_kit.core.script import BashScript, BashStep
from ai2_kit.core.util import load_yaml_files, merge_dict


def set_checkpoint_dir(path: str):
    """Set the checkpoint directory for resumable workflows."""
    checkpoint_service.set_checkpoint_dir(path)


__all__ = [
    "apply_checkpoint",
    "set_checkpoint_dir",
    "Artifact",
    "BashScript",
    "BashStep",
    "load_yaml_files",
    "merge_dict",
]
