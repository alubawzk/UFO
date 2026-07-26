#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

INPUT_DIR="${1:-humanoidverse/data/lafan1_mini3}"
MANIFEST_PATH="${2:-configs/data/lafan1_mini3_inference.yaml}"
DATASET_NAME="lafan1_mini3"

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "Input directory does not exist: ${INPUT_DIR}" >&2
  exit 1
fi

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_CMD=("${REPO_ROOT}/.venv/bin/python")
elif command -v uv >/dev/null 2>&1; then
  PYTHON_CMD=(uv run python)
else
  echo "Neither ${REPO_ROOT}/.venv/bin/python nor uv is available." >&2
  exit 1
fi

mapfile -d '' SOURCE_FILES < <(find "${INPUT_DIR}" -type f -name '*.pkl' -print0 | sort -z)
if (( ${#SOURCE_FILES[@]} == 0 )); then
  echo "No .pkl files found under: ${INPUT_DIR}" >&2
  exit 1
fi

echo "Converting ${#SOURCE_FILES[@]} full Mini3 motions from ${INPUT_DIR}"

"${PYTHON_CMD[@]}" -m humanoidverse.tools.data_build \
  --robot configs/robots/mini3.yaml \
  --source "${INPUT_DIR}" \
  --format robot_state_pkl \
  --name "${DATASET_NAME}" \
  --clip-seconds 10 \
  --stride-seconds 10 \
  --out "${MANIFEST_PATH}" \
  --rebuild-cache \
  --force

MANIFEST_NAME="${MANIFEST_PATH##*/}"
CACHE_NAME="${MANIFEST_NAME%.yaml}"
CACHE_NAME="${CACHE_NAME%.yml}"
CACHE_DIR="${REPO_ROOT}/cache/motion_data/${CACHE_NAME}"
FULL_PATH="${CACHE_DIR}/${DATASET_NAME}_full_ufo.pkl"
TRAIN_PATH="${CACHE_DIR}/${DATASET_NAME}_train_near10s_ufo.pkl"
if [[ "${MANIFEST_PATH}" = /* ]]; then
  MANIFEST_ABS="${MANIFEST_PATH}"
else
  MANIFEST_ABS="${REPO_ROOT}/${MANIFEST_PATH}"
fi

if [[ ! -f "${FULL_PATH}" || ! -f "${TRAIN_PATH}" ]]; then
  echo "Conversion finished without the expected output files in ${CACHE_DIR}" >&2
  exit 1
fi

echo
echo "Conversion complete."
echo "  Full inference data: ${FULL_PATH}"
echo "  Near-10-second training data: ${TRAIN_PATH}"
echo "  Data manifest: ${MANIFEST_ABS}"
echo
echo "Use the full file with:"
echo "  --data-path ${FULL_PATH}"
