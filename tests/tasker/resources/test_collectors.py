"""Test result collectors for training, exploration, and labeling."""

import numpy as np
from pathlib import Path
from catflow.tasker.collectors.train import (
    collect_train_results,
    _read_lcurve,
    estimate_trust_level,
)
from catflow.tasker.collectors.exploration import (
    collect_exploration_results,
    generate_shuffled_stats,
)
from catflow.tasker.collectors.labeling import (
    count_fp_candidates,
)


class TestTrainCollector:
    def test_read_lcurve(self, tmp_path):
        """Parse a valid lcurve.out file."""
        lcurve = tmp_path / "lcurve.out"
        content = "\n".join(
            f"{s} {s*0.1:.4f} {s*0.2:.4f} {s*0.3:.4f} {s*0.01:.4f} {s*0.02:.4f} {s*0.5:.4f}"
            for s in range(1, 101, 10)
        )
        lcurve.write_text(content)

        result = _read_lcurve(lcurve)
        assert "step" in result
        assert "rmse_f" in result
        assert "rmse_e" in result
        assert len(result["step"]) == 10

    def test_read_lcurve_not_found(self, tmp_path):
        """Non-existent lcurve.out returns empty results."""
        result = _read_lcurve(tmp_path / "nonexistent.out")
        assert result == {"step": [], "rmse_f": [], "rmse_e": []}

    def test_estimate_trust_level_from_error(self, tmp_path):
        """Trust level estimated from lcurve data."""
        Path.mkdir(tmp_path / "iter.000000" / "00.train" / "000", parents=True)
        lcurve = tmp_path / "iter.000000" / "00.train" / "000" / "lcurve.out"
        content = "\n".join(
            f"{s} 0.1 0.2 {s*0.01:.4f} 0.01 0.02 {s*0.03:.4f}"
            for s in range(1, 101, 10)
        )
        lcurve.write_text(content)

        lo, hi = estimate_trust_level(0, base_dir=str(tmp_path))
        assert lo > 0
        assert hi > lo
        assert pytest.approx(hi / lo, 0.1) == 3.0

    def test_collect_train_results_no_dir(self, tmp_path):
        """Missing train directory returns failed status."""
        result = collect_train_results(0, base_dir=str(tmp_path))
        assert result["status"] == "failed"


class TestExplorationCollector:
    def test_collect_exploration_results_no_dir(self, tmp_path):
        """Missing exploration directory returns empty."""
        result = collect_exploration_results(0, base_dir=str(tmp_path))
        assert result == {}

    def test_parse_model_devi(self, tmp_path):
        """Parse a valid model_devi.out file."""
        from catflow.tasker.collectors.exploration import _parse_model_devi

        mdf = tmp_path / "model_devi.out"
        content = "\n".join(
            f"{s*0.002:.4f} {s*0.01:.4f} {s*0.02:.4f} {s*0.03:.4f} {s*0.05:.4f} {s*0.1:.4f} {s*0.2:.4f}"
            for s in range(1, 51, 10)
        )
        mdf.write_text(content)

        result = _parse_model_devi(mdf)
        assert result is not None
        assert "step" in result
        assert "max_devi_f" in result
        assert len(result["max_devi_f"]) == 5

    def test_generate_shuffled_stats(self, tmp_path):
        """Shuffled stats files are created with correct categories."""
        results = {
            0: {
                "max_devi_f": np.array([0.05, 0.15, 0.35, 0.08, 0.20]),
                "task_names": ["a", "a", "a", "b", "b"],
                "frame_indices": [1, 2, 3, 1, 2],
            }
        }
        generate_shuffled_stats(
            results,
            f_trust_lo=0.1,
            f_trust_hi=0.3,
            base_dir=str(tmp_path),
            iter_index=0,
        )

        fp_dir = tmp_path / "iter.000000" / "02.fp"
        candidate_file = fp_dir / "candidate.shuffled.000.out"
        assert candidate_file.exists()
        content = candidate_file.read_text().strip()
        # Frames with 0.15 and 0.20 should be candidates
        assert "a 2" in content
        assert "b 2" in content


class TestLabelingCollector:
    def test_count_fp_candidates(self, tmp_path):
        """Count candidates from shuffled files."""
        fp_dir = tmp_path / "iter.000000" / "02.fp"
        fp_dir.mkdir(parents=True)

        (fp_dir / "candidate.shuffled.000.out").write_text("a 1\nb 2\nc 3\n")
        counts = count_fp_candidates(0, base_dir=str(tmp_path))
        assert counts == {0: 3}

    def test_count_fp_candidates_multiple_systems(self, tmp_path):
        """Multiple system indices are counted separately."""
        fp_dir = tmp_path / "iter.000000" / "02.fp"
        fp_dir.mkdir(parents=True)
        (fp_dir / "candidate.shuffled.000.out").write_text("a 1\nb 2\n")
        (fp_dir / "candidate.shuffled.001.out").write_text("c 3\n")
        counts = count_fp_candidates(0, base_dir=str(tmp_path))
        assert counts == {0: 2, 1: 1}

    def test_count_fp_candidates_no_dir(self, tmp_path):
        """No FP directory returns empty."""
        counts = count_fp_candidates(0, base_dir=str(tmp_path))
        assert counts == {}


import pytest  # noqa: needed for pytest.approx
