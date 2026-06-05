"""Training result collector.

Replaces dpgen.generator.run.post_train.
Reads lcurve.out, collects frozen models, and returns training metrics.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from glob import glob

from catflow.utils import logger


def collect_train_results(iter_index: int, base_dir: str = ".") -> Dict:
    """Collect DeePMD training results from an iteration.

    Args:
        iter_index: Iteration index (0-based).
        base_dir: Base working directory.

    Returns:
        Dict with:
            - "model_files": List of paths to frozen model files.
            - "lcurve": Dict with "step", "rmse_f", "rmse_e" arrays.
            - "n_models": Number of trained models.
            - "status": "completed" or "failed".
    """
    iter_dir = Path(base_dir) / f"iter.{str(iter_index).zfill(6)}"
    train_dir = iter_dir / "00.train"

    if not train_dir.exists():
        logger.error(f"Training directory not found: {train_dir}")
        return {"status": "failed", "error": "train_dir_not_found"}

    # Collect frozen model files
    model_files = []
    model_dirs = sorted(glob(str(train_dir / "*/")))
    for md in model_dirs:
        for model_name in ["frozen_model.pb", "graph.pb"]:
            model_path = Path(md) / model_name
            if model_path.exists():
                model_files.append(str(model_path.resolve()))
                break

    # Read lcurve.out from first model
    lcurve_data = _read_lcurve(train_dir / "000" / "lcurve.out")

    result = {
        "model_files": model_files,
        "lcurve": lcurve_data,
        "n_models": len(model_files),
        "status": "completed" if model_files else "failed",
    }

    logger.info(
        f"[Collector] Training iter.{iter_index}: "
        f"{len(model_files)} models, "
        f"final rmse_f={lcurve_data.get('rmse_f', [-1])[-1]:.4f}"
    )

    return result


def _read_lcurve(lcurve_path: Path) -> Dict:
    """Parse DeePMD lcurve.out file.

    Format: step, rmse_val, rmse_trn, rmse_e_val, rmse_e_trn, rmse_f_val, rmse_f_trn
    """
    if not lcurve_path.exists():
        logger.warning(f"lcurve.out not found: {lcurve_path}")
        return {"step": [], "rmse_f": [], "rmse_e": []}

    try:
        data = np.loadtxt(lcurve_path)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return {
            "step": data[:, 0].tolist(),
            "rmse_f": data[:, -1].tolist(),  # rmse_f_trn (last column)
            "rmse_e": data[:, 3].tolist(),   # rmse_e_trn
        }
    except Exception as e:
        logger.warning(f"Failed to parse lcurve.out: {e}")
        return {"step": [], "rmse_f": [], "rmse_e": []}


def estimate_trust_level(iter_index: int, base_dir: str = ".") -> Tuple[float, float]:
    """Estimate trust level from training error.

    Returns:
        Tuple of (f_trust_lo, f_trust_hi).
    """
    lcurve = _read_lcurve(
        Path(base_dir) / f"iter.{str(iter_index).zfill(6)}" / "00.train" / "000" / "lcurve.out"
    )
    rmse_f = lcurve.get("rmse_f", [0.2])
    if not rmse_f:
        rmse_f = [0.2]
    mean_err = np.mean(rmse_f[-max(1, len(rmse_f)//10):])
    return round(mean_err * 0.9, 2), round(mean_err * 3.0, 2)
