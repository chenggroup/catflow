#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "Starting DeePMD training in $PWD"
echo "Using GPU: ${CUDA_VISIBLE_DEVICES:-none}"

# Run DeePMD training (adjust command as needed for your system)
if [ -f "input.json" ]; then
    dp train input.json
    dp freeze -o frozen_model.pb
    dp compress -i frozen_model.pb -o graph.pb -t input.json 2>/dev/null || \
        cp frozen_model.pb graph.pb
    echo "Model frozen: frozen_model.pb / graph.pb"
else
    echo "ERROR: input.json not found"
    exit 1
fi

echo "Training complete: $PWD"
