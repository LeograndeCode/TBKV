#!/bin/bash
# Helper script to run eventful-transformer scripts with proper environment setup

# Usage: ./run_script.sh <script_path> <config_name> [additional args...]
# Example: ./run_script.sh ./scripts/evaluate/vivit_kinetics400.py base

set -e

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <script_path> <config_name> [additional_args...]"
    echo "Example: $0 ./scripts/evaluate/vivit_kinetics400.py base"
    exit 1
fi

SCRIPT_PATH="$1"
CONFIG_NAME="$2"
shift 2
EXTRA_ARGS="$@"

# Get the directory of this script (should be the repo root)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate eventful-transformer

# Set PYTHONPATH to include repo directory
export PYTHONPATH="${PYTHONPATH}:${REPO_DIR}"

# Change to repo directory and run the script
cd "${REPO_DIR}"
"${SCRIPT_PATH}" "${CONFIG_NAME}" ${EXTRA_ARGS}
