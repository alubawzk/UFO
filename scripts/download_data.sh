#!/usr/bin/env bash
set -euo pipefail

DATASET_REPO="${UFO_DATASET_REPO:-xuewang/UFO-MotionData}"
DATASET_REVISION="${UFO_DATASET_REVISION:-main}"
DATASET_FILE="g1/lafan_29dof_10s-clipped.pkl"
DEST="humanoidverse/data/lafan_29dof_10s-clipped.pkl"
EXPECTED_SHA256="7f5aa36957808ee2e972472b18add8510533742710ba312d8b8c6e6014f1c010"

usage() {
  cat <<USAGE
Usage: bash scripts/download_data.sh [g1_lafan]

Downloads the default G1 LaFAN training motion data from Hugging Face:
  https://huggingface.co/datasets/${DATASET_REPO}

Environment overrides:
  UFO_DATASET_REPO=${DATASET_REPO}
  UFO_DATASET_REVISION=${DATASET_REVISION}
USAGE
}

case "${1:-g1_lafan}" in
  g1_lafan|lafan|g1)
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown dataset: $1" >&2
    usage >&2
    exit 2
    ;;
esac

if [[ -f "${DEST}" ]]; then
  actual="$(sha256sum "${DEST}" | awk '{print $1}')"
  if [[ "${actual}" == "${EXPECTED_SHA256}" ]]; then
    echo "Data already present: ${DEST}"
    exit 0
  fi
  echo "Existing ${DEST} has unexpected sha256: ${actual}" >&2
  echo "Re-downloading." >&2
fi

mkdir -p "$(dirname "${DEST}")"
tmpdir="$(mktemp -d)"
cleanup() {
  rm -rf "${tmpdir}"
}
trap cleanup EXIT

if command -v hf >/dev/null 2>&1; then
  hf download "${DATASET_REPO}" \
    "${DATASET_FILE}" \
    --repo-type dataset \
    --revision "${DATASET_REVISION}" \
    --local-dir "${tmpdir}"
elif python3 - <<'PYTEST' >/dev/null 2>&1
import huggingface_hub
PYTEST
then
  python3 - "${DATASET_REPO}" "${DATASET_FILE}" "${DATASET_REVISION}" "${tmpdir}" <<'PYDL'
import sys
from huggingface_hub import hf_hub_download

repo_id, filename, revision, local_dir = sys.argv[1:]
path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    repo_type="dataset",
    revision=revision,
    local_dir=local_dir,
)
print(path)
PYDL
else
  cat >&2 <<'EOF'
Missing Hugging Face downloader.
Install one of the following and rerun:
  python -m pip install -U huggingface_hub
or
  uv tool install huggingface_hub
EOF
  exit 1
fi

src="${tmpdir}/${DATASET_FILE}"
if [[ ! -f "${src}" ]]; then
  echo "Download failed: ${src} not found" >&2
  exit 1
fi

actual="$(sha256sum "${src}" | awk '{print $1}')"
if [[ "${actual}" != "${EXPECTED_SHA256}" ]]; then
  echo "sha256 mismatch for downloaded data" >&2
  echo "expected: ${EXPECTED_SHA256}" >&2
  echo "actual:   ${actual}" >&2
  exit 1
fi

cp "${src}" "${DEST}"
echo "Downloaded ${DEST}"
ls -lh "${DEST}"
