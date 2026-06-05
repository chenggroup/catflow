#!/bin/bash
set -e

# TESLA + oh-my-batch workflow: outer loop
# Runs multiple iterations of Training → Exploration → Labeling

CONFIG_DIR=./00-config
WORK_DIR=./10-workdir

# Ensure dependencies
pip install "oh-my-batch>=0.4.8" 2>/dev/null || true

# Run initial setup
bash 01-workflow/setup.sh

# Active learning iterations
for ITER_NAME in 000 001 002; do
    echo ""
    echo "========================================"
    echo "Starting iteration ${ITER_NAME}"
    echo "========================================"

    ITER_NAME="${ITER_NAME}" bash 01-workflow/iter.sh

    echo "Iteration ${ITER_NAME} completed."
done

echo ""
echo "All iterations completed successfully."
