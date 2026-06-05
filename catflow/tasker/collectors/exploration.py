"""Exploration result collector.

Replaces dpgen.generator.run.post_model_devi.
Parses model_devi.out files, generates shuffled statistics for FP selection.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from glob import glob

from catflow.utils import logger


def collect_exploration_results(iter_index: int, base_dir: str = ".") -> Dict:
    """Collect and analyze model deviation results from exploration.

    Reads model_devi.out from each task directory and computes:
    - Max force deviation per frame
    - Classification into accurate/candidate/failed

    Args:
        iter_index: Iteration index.
        base_dir: Base working directory.

    Returns:
        Dict with per-system statistics.
    """
    iter_dir = Path(base_dir) / f"iter.{str(iter_index).zfill(6)}"
    exp_dir = iter_dir / "01.model_devi"

    if not exp_dir.exists():
        logger.error(f"Exploration directory not found: {exp_dir}")
        return {}

    results = {}
    task_dirs = sorted(glob(str(exp_dir / "task.*/")))

    for task_dir in task_dirs:
        td = Path(task_dir)
        model_devi_file = td / "model_devi.out"
        if not model_devi_file.exists():
            continue

        # Parse task name for sys_idx
        task_name = td.name  # e.g. "task.0.3"
        parts = task_name.split(".")
        if len(parts) >= 3:
            sys_idx = int(parts[1])
        else:
            continue

        # Parse model_devi.out
        devi_data = _parse_model_devi(model_devi_file)
        if devi_data is None:
            continue

        if sys_idx not in results:
            results[sys_idx] = {
                "max_devi_f": [],
                "task_names": [],
                "frame_indices": [],
            }

        results[sys_idx]["max_devi_f"].extend(devi_data["max_devi_f"])
        results[sys_idx]["task_names"].extend([task_name] * len(devi_data["max_devi_f"]))
        results[sys_idx]["frame_indices"].extend(
            range(len(devi_data["max_devi_f"]))
        )

    # Compute per-system statistics
    for sys_idx, data in results.items():
        data["max_devi_f"] = np.array(data["max_devi_f"])

    logger.info(
        f"[Collector] Exploration iter.{iter_index}: "
        f"{len(results)} systems analyzed"
    )
    return results


def _parse_model_devi(filepath: Path) -> Optional[Dict]:
    """Parse a LAMMPS model_devi.out file.

    Columns: step, rmse_f, rmse_e, rmse_v, max_devi_f, max_devi_e, max_devi_v
    """
    try:
        data = np.loadtxt(filepath)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape[1] < 5:
            return None

        return {
            "step": data[:, 0].tolist(),
            "max_devi_f": data[:, 4].tolist(),  # max_devi_f column
        }
    except Exception as e:
        logger.warning(f"Failed to parse {filepath}: {e}")
        return None


def generate_shuffled_stats(
    results: Dict[int, Dict],
    f_trust_lo: float = 0.1,
    f_trust_hi: float = 0.3,
    base_dir: str = ".",
    iter_index: int = 0,
) -> Dict[int, Tuple[List, List, List]]:
    """Generate shuffled statistics files for FP selection.

    Mimics dpgen's candidate.shuffled.*.out, rest_accurate.shuffled.*.out,
    and rest_failed.shuffled.*.out files.

    Args:
        results: Output from collect_exploration_results().
        f_trust_lo: Lower trust level for force deviation.
        f_trust_hi: Upper trust level for force deviation.
        base_dir: Base working directory.
        iter_index: Iteration index.

    Returns:
        Dict mapping sys_idx to (candidates, accurate, failed) lists.
    """
    iter_dir = Path(base_dir) / f"iter.{str(iter_index).zfill(6)}"
    fp_dir = iter_dir / "02.fp"
    fp_dir.mkdir(parents=True, exist_ok=True)

    stats = {}
    for sys_idx, data in results.items():
        devi_f = data["max_devi_f"]

        candidates = []
        accurate = []
        failed = []

        for i, (task_name, frame_idx) in enumerate(
            zip(data["task_names"], data["frame_indices"])
        ):
            if i >= len(devi_f):
                break
            df = devi_f[i]
            if df < f_trust_lo:
                accurate.append(f"{task_name} {frame_idx}")
            elif df > f_trust_hi:
                failed.append(f"{task_name} {frame_idx}")
            else:
                candidates.append(f"{task_name} {frame_idx}")

        # Write shuffled files
        sys_str = str(sys_idx).zfill(3)

        with open(fp_dir / f"candidate.shuffled.{sys_str}.out", "w") as f:
            f.write("\n".join(candidates) + "\n")

        with open(fp_dir / f"rest_accurate.shuffled.{sys_str}.out", "w") as f:
            f.write("\n".join(accurate) + "\n")

        with open(fp_dir / f"rest_failed.shuffled.{sys_str}.out", "w") as f:
            f.write("\n".join(failed) + "\n")

        stats[sys_idx] = (candidates, accurate, failed)

        logger.info(
            f"[Collector] Sys {sys_idx}: "
            f"{len(candidates)} candidates, "
            f"{len(accurate)} accurate, "
            f"{len(failed)} failed"
        )

    return stats
