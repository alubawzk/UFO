#!/usr/bin/env bash
set -euo pipefail

PICO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UFO_DIR="$(cd "${PICO_DIR}/.." && pwd)"
BUILD_DIR="${PICO_DIR}/.build"
DOWNLOAD_DIR="${PICO_DIR}/.downloads"
NATIVE_DIR="${PICO_DIR}/native"
XR_SOURCE_DIR="${BUILD_DIR}/XRoboToolkit-PC-Service"
XR_SDK_DIR="${XR_SOURCE_DIR}/RoboticsService/PXREARobotSDK"
XR_REPOSITORY="https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git"
XR_VERSION="v1.0.0"
UV_VERSION="0.11.28"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "This installer currently supports x86_64 Linux only." >&2
  exit 1
fi

if [[ ! -f /etc/os-release ]]; then
  echo "Cannot identify this Linux distribution (/etc/os-release is missing)." >&2
  exit 1
fi
. /etc/os-release
case "${VERSION_ID}" in
  22.04|24.04) UBUNTU_VERSION="${VERSION_ID}" ;;
  *)
    echo "XRoboToolkit v1.0.0 provides packages for Ubuntu 22.04/24.04; found ${PRETTY_NAME}." >&2
    exit 1
    ;;
esac

mkdir -p "${BUILD_DIR}" "${DOWNLOAD_DIR}" "${NATIVE_DIR}"

echo "[1/5] Installing system build tools"
sudo apt-get update
sudo apt-get install -y build-essential cmake curl git git-lfs

echo "[2/5] Installing the official XRoboToolkit PC Service ${XR_VERSION}"
if [[ ! -f /opt/apps/roboticsservice/runService.sh ]]; then
  DEB_NAME="XRoboToolkit_PC_Service_1.0.0_ubuntu_${UBUNTU_VERSION}_amd64.deb"
  DEB_PATH="${DOWNLOAD_DIR}/${DEB_NAME}"
  curl -fL \
    "https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/${XR_VERSION}/${DEB_NAME}" \
    -o "${DEB_PATH}"
  sudo dpkg -i "${DEB_PATH}" || sudo apt-get install -f -y
else
  echo "PC Service already present; skipping .deb installation"
fi

echo "[3/5] Synchronizing UFO Python dependencies"
cd "${UFO_DIR}"
git lfs install --local
SMPLX_MODEL="${PICO_DIR}/smplx/SMPLX_NEUTRAL.pkl"
if [[ ! -f "${SMPLX_MODEL}" ]] || [[ "$(stat -c %s "${SMPLX_MODEL}")" -lt 100000000 ]]; then
  echo "Fetching the Git LFS SMPL-X model"
  git lfs pull --include="pico_sim2sim/smplx/SMPLX_NEUTRAL.pkl"
fi
if [[ ! -f "${SMPLX_MODEL}" ]] || [[ "$(stat -c %s "${SMPLX_MODEL}")" -lt 100000000 ]]; then
  echo "Missing real SMPL-X model: ${SMPLX_MODEL}" >&2
  echo "Check Git LFS access and the SMPL-X model license before continuing." >&2
  exit 1
fi
if command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
else
  echo "uv is not installed; bootstrapping uv ${UV_VERSION} under pico_sim2sim/.tools"
  UV_INSTALLER="${DOWNLOAD_DIR}/uv-${UV_VERSION}-install.sh"
  curl -fLsS "https://astral.sh/uv/${UV_VERSION}/install.sh" -o "${UV_INSTALLER}"
  UV_UNMANAGED_INSTALL="${PICO_DIR}/.tools" sh "${UV_INSTALLER}"
  UV_BIN="${PICO_DIR}/.tools/uv"
fi
"${UV_BIN}" sync --extra pico-teleop

echo "[4/5] Building the official PXREA SDK ${XR_VERSION}"
if [[ ! -d "${XR_SOURCE_DIR}/.git" ]]; then
  git clone --depth 1 --branch "${XR_VERSION}" "${XR_REPOSITORY}" "${XR_SOURCE_DIR}"
fi
(
  cd "${XR_SDK_DIR}"
  bash build.sh
)

echo "[5/5] Building the repository-local Python binding"
PYBIND11_CMAKE_DIR="$("${UFO_DIR}/.venv/bin/python" -m pybind11 --cmakedir)"
cmake \
  -S "${PICO_DIR}/xrobotoolkit_binding" \
  -B "${BUILD_DIR}/python-binding" \
  -DPXREA_SDK_ROOT="${XR_SDK_DIR}" \
  -Dpybind11_DIR="${PYBIND11_CMAKE_DIR}" \
  -DPython_EXECUTABLE="${UFO_DIR}/.venv/bin/python" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="${NATIVE_DIR}"
cmake --build "${BUILD_DIR}/python-binding" --parallel
cp "${XR_SDK_DIR}/build/libPXREARobotSDK.so" "${NATIVE_DIR}/"

echo
echo "PICO sim2sim installation complete."
echo "Run diagnostics: ${UFO_DIR}/.venv/bin/python -m pico_sim2sim.doctor"
