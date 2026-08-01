#!/usr/bin/env bash
# Shared setup sourced by every run/NN_*.sh script.
#
# Resolves the repository root, activates the conda environment, and provides
# the `run` helper (echo the command, then execute it with unbuffered output)
# and the `skip_if_done` guard so a re-run does not repeat finished work.
#
# Override points (export before calling a script):
#   CONDA_SH   path to conda's profile.d/conda.sh   (auto-detected otherwise)
#   ENV_NAME   conda environment name               (default: eventful-transformer)
#   FORCE=1    re-run steps whose outputs already exist

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME="${ENV_NAME:-eventful-transformer}"
FORCE="${FORCE:-0}"

# ---------------------------------------------------------------- conda ----
if [ -z "${CONDA_SH:-}" ]; then
    for candidate in \
        "$HOME/miniconda3/etc/profile.d/conda.sh" \
        "$HOME/anaconda3/etc/profile.d/conda.sh" \
        "/opt/conda/etc/profile.d/conda.sh" \
        "${CONDA_EXE:-}/../../etc/profile.d/conda.sh"; do
        [ -f "$candidate" ] && CONDA_SH="$candidate" && break
    done
fi

if [ -n "${CONDA_SH:-}" ] && [ -f "$CONDA_SH" ]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    conda activate "$ENV_NAME" || {
        echo "ERROR: could not activate conda env '$ENV_NAME'." >&2
        echo "       Create it with:  conda env create -f environment.yml" >&2
        exit 1
    }
else
    echo "ERROR: could not find conda.sh. Set CONDA_SH=/path/to/etc/profile.d/conda.sh" >&2
    exit 1
fi

cd "$REPO" || exit 1

# Every evaluation script is invoked from the archive root and writes under
# results/; the cd above keeps that contract explicit so a stray caller cwd
# cannot silently redirect outputs elsewhere.
export PYTHONUNBUFFERED=1

# ---------------------------------------------------------------- helpers ----
run() {
    echo ""
    echo "### $*"
    echo ""
    "$@"
}

# skip_if_done <marker-path> <human label>
# Returns 0 (proceed) or 1 (skip). A marker is the run's output.txt/results.txt.
skip_if_done() {
    local marker="$1" label="$2"
    if [ "$FORCE" != "1" ] && [ -s "$marker" ]; then
        echo "--- SKIP (already present): $label"
        echo "    $marker"
        return 1
    fi
    return 0
}

banner() {
    echo ""
    echo "============================================================================"
    echo "  $*"
    echo "============================================================================"
}

gpu_busy_warning() {
    # Latency and peak memory are only valid on an idle GPU. Warn, do not block.
    if command -v nvidia-smi >/dev/null 2>&1; then
        local n
        n="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)"
        if [ "${n:-0}" -gt 0 ]; then
            echo ""
            echo "WARNING: $n process(es) already using the GPU."
            echo "         mAP/Top-1/GFLOPs are unaffected, but the latency and peak-memory"
            echo "         columns of Table 1 will be wrong. Run on an idle device."
            echo ""
        fi
    fi
}

VID_RES="results/evaluate/vitdet_vid"
K400_RES="results/evaluate/vivit_kinetics400"
