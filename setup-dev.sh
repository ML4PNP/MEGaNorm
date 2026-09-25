#!/usr/bin/env bash

set -euo pipefail

# MEGaNorm joint development environment bootstrap.
#
# Expected layout:
#
# parent/
#   MEGaNorm/
#     setup-dev.sh
#   pcntkdev/
#     PCNtoolkit/
#
# MEGaNorm may use a developer-selected branch. The default is "dev".
# PCNtoolkit must use the "integration" branch.
#
# The script validates Git state but intentionally does not switch branches
# or pull changes.

ENV_NAME="meganorm-dev"
PYTHON_VERSION="3.12"

MEGANORM_BRANCH="${1:-dev}"

PCNTOOLKIT_BRANCH="integration"
PCNTOOLKIT_DEV_ROOT="pcntkdev"
PCNTOOLKIT_REPO_NAME="PCNtoolkit"

if [[ "${MEGANORM_BRANCH}" == "-h" || "${MEGANORM_BRANCH}" == "--help" ]]; then
    cat <<'EOF'
Usage:
  ./setup-dev.sh [MEGANORM_BRANCH]

Set up the joint MEGaNorm and PCNtoolkit development environment.

Arguments:
  MEGANORM_BRANCH   MEGaNorm branch to use for development.
                    Optional. Defaults to "dev".

Requirements:
  - MEGaNorm must already be checked out on the requested branch.
  - PCNtoolkit must already be checked out on the "integration" branch.
  - The expected directory layout is:

      parent/
        MEGaNorm/
        pcntkdev/
          PCNtoolkit/

Examples:
  ./setup-dev.sh
  ./setup-dev.sh dev
  ./setup-dev.sh dev_meg
  ./setup-dev.sh feature/new-workflow

The script validates Git branches but does not switch branches or pull changes.
EOF
    exit 0
fi

if [[ "$#" -gt 1 ]]; then
    echo "ERROR: Too many arguments."
    echo "Run './setup-dev.sh --help' for usage."
    exit 2
fi

MEGANORM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PCNTOOLKIT_DIR="$(
    cd "${MEGANORM_DIR}/../${PCNTOOLKIT_DEV_ROOT}/${PCNTOOLKIT_REPO_NAME}" \
        2>/dev/null && pwd || true
)"

echo
echo "MEGaNorm development environment setup"
echo "======================================="
echo
echo "MEGaNorm branch:   ${MEGANORM_BRANCH}"
echo "PCNtoolkit branch: ${PCNTOOLKIT_BRANCH}"
echo "Conda environment: ${ENV_NAME}"
echo

command -v git >/dev/null 2>&1 || {
    echo "ERROR: git is not available."
    exit 1
}

command -v conda >/dev/null 2>&1 || {
    echo "ERROR: conda is not available. Install Conda or Miniforge and try again."
    exit 1
}

if ! git -C "${MEGANORM_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "ERROR: ${MEGANORM_DIR} is not a Git repository."
    exit 1
fi

CURRENT_MEGANORM_BRANCH="$(git -C "${MEGANORM_DIR}" branch --show-current)"

if [[ "${CURRENT_MEGANORM_BRANCH}" != "${MEGANORM_BRANCH}" ]]; then
    echo "ERROR: MEGaNorm is on branch '${CURRENT_MEGANORM_BRANCH}'."
    echo "Requested branch: '${MEGANORM_BRANCH}'."
    echo
    echo "Switch manually with:"
    echo "  git -C \"${MEGANORM_DIR}\" switch \"${MEGANORM_BRANCH}\""
    exit 1
fi

echo "MEGaNorm branch verified: ${CURRENT_MEGANORM_BRANCH}"

if [[ -z "${PCNTOOLKIT_DIR}" ]]; then
    echo "ERROR: PCNtoolkit repository was not found at:"
    echo "  ${MEGANORM_DIR}/../${PCNTOOLKIT_DEV_ROOT}/${PCNTOOLKIT_REPO_NAME}"
    echo
    echo "Expected directory layout:"
    echo "  parent/"
    echo "    MEGaNorm/"
    echo "    ${PCNTOOLKIT_DEV_ROOT}/"
    echo "      ${PCNTOOLKIT_REPO_NAME}/"
    exit 1
fi

if ! git -C "${PCNTOOLKIT_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "ERROR: ${PCNTOOLKIT_DIR} is not a Git repository."
    exit 1
fi

CURRENT_PCNTOOLKIT_BRANCH="$(git -C "${PCNTOOLKIT_DIR}" branch --show-current)"

if [[ "${CURRENT_PCNTOOLKIT_BRANCH}" != "${PCNTOOLKIT_BRANCH}" ]]; then
    echo "ERROR: PCNtoolkit is on branch '${CURRENT_PCNTOOLKIT_BRANCH}'."
    echo "Required branch: '${PCNTOOLKIT_BRANCH}'."
    echo
    echo "Switch manually with:"
    echo "  git -C \"${PCNTOOLKIT_DIR}\" switch \"${PCNTOOLKIT_BRANCH}\""
    exit 1
fi

echo "PCNtoolkit branch verified: ${CURRENT_PCNTOOLKIT_BRANCH}"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Conda environment '${ENV_NAME}' already exists."
else
    echo
    echo "Creating Conda environment '${ENV_NAME}'..."

    conda create \
        --channel=conda-forge \
        --strict-channel-priority \
        --name "${ENV_NAME}" \
        "python=${PYTHON_VERSION}" \
        --yes

    echo "Conda environment created."
fi

echo
echo "Updating pip..."

conda run -n "${ENV_NAME}" \
    python -m pip install --upgrade pip

echo
echo "Installing PCNtoolkit from the local integration branch..."

conda run -n "${ENV_NAME}" \
    python -m pip install -e "${PCNTOOLKIT_DIR}"

echo
echo "Installing MEGaNorm from the local '${MEGANORM_BRANCH}' branch..."

conda run -n "${ENV_NAME}" \
    python -m pip install -e "${MEGANORM_DIR}[dev]"

echo
echo "Verifying development environment..."

conda run -n "${ENV_NAME}" python - <<'PY'
import sys
import meganorm
import pcntoolkit

print(f"Python:     {sys.version.split()[0]}")
print(f"MEGaNorm:   {meganorm.__file__}")
print(f"PCNtoolkit: {pcntoolkit.__file__}")
PY

echo
echo "======================================="
echo "Development environment is ready."
echo
echo "Activate it with:"
echo
echo "  conda activate ${ENV_NAME}"
echo
