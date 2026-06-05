"""Labeling result collector.

Replaces dpgen.generator.run.post_fp.
Parses CP2K/VASP output, converts to dpdata format.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from glob import glob

from catflow.utils import logger


def collect_labeling_results(
    iter_index: int,
    base_dir: str = ".",
    output_format: str = "deepmd/npy",
) -> Dict:
    """Collect FP labeling results and convert to training dataset.

    Args:
        iter_index: Iteration index.
        base_dir: Base working directory.
        output_format: dpdata output format (default: deepmd/npy).

    Returns:
        Dict with:
            - "data_path": Path to the collected dataset.
            - "n_frames": Number of successfully labeled frames.
            - "status": "completed" or "failed".
    """
    import dpdata

    iter_dir = Path(base_dir) / f"iter.{str(iter_index).zfill(6)}"
    fp_dir = iter_dir / "02.fp"
    data_dir = fp_dir / "data"

    if not fp_dir.exists():
        logger.error(f"FP directory not found: {fp_dir}")
        return {"status": "failed", "error": "fp_dir_not_found"}

    # Detect FP software from params
    task_dirs = sorted(glob(str(fp_dir / "task.*/")))
    if not task_dirs:
        logger.warning(f"No FP tasks found in {fp_dir}")
        return {"status": "failed", "error": "no_tasks"}

    # Determine format from available output files
    first_task = Path(task_dirs[0])
    fmt, style = _detect_fp_format(first_task)

    # Collect all labeled systems
    all_sys = None
    n_frames = 0

    for task_dir in task_dirs:
        td = Path(task_dir)
        try:
            if fmt == "cp2k/output":
                sys = dpdata.LabeledSystem(str(td / "output"), fmt=fmt)
            elif fmt == "vasp/outcar":
                sys = dpdata.LabeledSystem(str(td / "OUTCAR"), fmt=fmt)
            elif fmt == "vasp/vasprun":
                sys = dpdata.LabeledSystem(str(td / "vasprun.xml"), fmt=fmt)
            else:
                continue

            if len(sys) > 0:
                if all_sys is None:
                    all_sys = sys
                else:
                    all_sys.append(sys)
                n_frames += len(sys)

        except Exception as e:
            logger.warning(f"Failed to collect {task_dir}: {e}")

    if all_sys is None or n_frames == 0:
        logger.warning(f"No labeled data collected from {fp_dir}")
        return {"status": "failed", "error": "no_data"}

    # Write to output directory
    data_dir.mkdir(parents=True, exist_ok=True)
    all_sys.to_deepmd_npy(str(data_dir))

    # Copy type_map.raw if exists
    init_data = Path(base_dir) / "iter.000000" / "00.train" / "data.init" / "type_map.raw"
    if init_data.exists():
        shutil.copy(init_data, data_dir / "type_map.raw")

    logger.info(
        f"[Collector] FP iter.{iter_index}: "
        f"{n_frames} frames collected to {data_dir}"
    )

    return {
        "data_path": str(data_dir),
        "n_frames": n_frames,
        "status": "completed",
    }


def _detect_fp_format(task_dir: Path) -> tuple:
    """Detect FP output format from available files."""
    if (task_dir / "OUTCAR").exists():
        return "vasp/outcar"
    if (task_dir / "vasprun.xml").exists():
        return "vasp/vasprun"
    if (task_dir / "output").exists():
        return "cp2k/output"
    return "unknown"


def count_fp_candidates(iter_index: int, base_dir: str = ".") -> Dict[int, int]:
    """Count candidate structures available for labeling.

    Args:
        iter_index: Iteration index.
        base_dir: Base working directory.

    Returns:
        Dict mapping sys_idx to candidate count.
    """
    iter_dir = Path(base_dir) / f"iter.{str(iter_index).zfill(6)}"
    fp_dir = iter_dir / "02.fp"

    if not fp_dir.exists():
        return {}

    counts = {}
    for cand_file in sorted(glob(str(fp_dir / "candidate.shuffled.*.out"))):
        parts = Path(cand_file).stem.split(".")
        if len(parts) >= 3:
            sys_idx = int(parts[-2])
            with open(cand_file) as f:
                counts[sys_idx] = sum(1 for line in f if line.strip())

    return counts
