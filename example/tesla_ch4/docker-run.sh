#!/bin/bash
set -e

# Run CH4 validation case inside deepmodeling/deepmd-kit Docker container
# This provides real dp (DeePMD train) and lmp (LAMMPS) commands

REGISTRY="${REGISTRY:-docker.io}"
IMAGE="${REGISTRY}/deepmodeling/deepmd-kit:3.1.3_cpu"
WORK_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Pulling image: ${IMAGE}"
docker pull "${IMAGE}"

echo ""
echo "Running CH4 validation inside container..."
echo "=========================================="

docker run --rm \
    -v "${WORK_DIR}:/workspace" \
    -w /workspace \
    "${IMAGE}" \
    bash -c '
        set -e
        echo "=== Container Info ==="
        echo "Image: '${IMAGE}'"
        echo "dp: $(command -v dp || echo NOT_FOUND)"
        echo "lmp: $(command -v lmp || echo NOT_FOUND)"
        echo "python: $(python3 --version)"
        echo ""

        # Install oh-my-batch (no dpgen needed)
        pip install oh-my-batch -q 2>/dev/null

        # Clean previous run
        rm -rf 10-workdir

        # Run the validation
        bash run.sh
    '
