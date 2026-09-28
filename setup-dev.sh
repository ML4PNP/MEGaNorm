#!/usr/bin/env bash

set -euo pipefail

# MEGaNorm Development Environment bootstrap.
#
# Expected layout:
#
# parent/
#   MEGaNorm/
#     setup-dev.sh
#   pcntkdev/
#     PCNtoolkit/
#
# Usage:
#   ./setup-dev.sh ENV_NAME MEGANORM_BRANCH [--editable|--shared]
#
# PCNtoolkit always uses the "integration" branch.

PYTHON_VERSION="3.12"

PCNTOOLKIT_BRANCH="integration"
PCNTOOLKIT_DEV_ROOT="pcntkdev"
PCNTOOLKIT_REPO_NAME="PCNtoolkit"

INSTALL_MODE="editable"

show_help() {
    cat <<'EOF'
Usage:
  ./setup-dev.sh ENV_NAME MEGANORM_BRANCH [--editable|--shared]

Set up or refresh the MEGaNorm Development Environment.

Required arguments:
  ENV_NAME           Name of the Conda environment to create or refresh.
  MEGANORM_BRANCH    MEGaNorm branch to use for development.

Options:
  --editable         Install MEGaNorm and PCNtoolkit in editable mode.
                     This is the default and is recommended for individual
                     developers.

  --shared           Install MEGaNorm and PCNtoolkit normally into the Conda
                     environment. Use this when the environment is shared by
                     users who do not have access to the source checkouts.

  -h, --help         Show this help message.

Expected directory layout:

  parent/
    MEGaNorm/
      setup-dev.sh
    pcntkdev/
      PCNtoolkit/

Examples:
  ./setup-dev.sh meganorm-dev-meg dev_meg
  ./setup-dev.sh meganorm-dev-meg dev_meg --editable
  ./setup-dev.sh meganorm-shared dev --shared
EOF
}

if [[ "$#" -eq 0 ]]; then
    show_help
    exit 2
fi

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    exit 0
fi

if [[ "$#" -lt 2 ]]; then
    echo "ERROR: ENV_NAME and MEGANORM_BRANCH are required."
    echo
    show_help
    exit 2
fi

ENV_NAME="$1"
MEGANORM_BRANCH="$2"
shift 2

if [[ "${ENV_NAME}" == -* ]]; then
    echo "ERROR: ENV_NAME must be the first argument."
    echo
    show_help
    exit 2
fi

if [[ "${MEGANORM_BRANCH}" == -* ]]; then
    echo "ERROR: MEGANORM_BRANCH must be the second argument."
    echo
    show_help
    exit 2
fi

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --editable)
            INSTALL_MODE="editable"
            ;;
        --shared)
            INSTALL_MODE="shared"
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option '$1'."
            echo
            show_help
            exit 2
            ;;
    esac
    shift
done

MEGANORM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PCNTOOLKIT_DIR="$(
    cd "${MEGANORM_DIR}/../${PCNTOOLKIT_DEV_ROOT}/${PCNTOOLKIT_REPO_NAME}" \
        2>/dev/null && pwd || true
)"

echo
echo "MEGaNorm Development Environment setup"
echo "======================================="
echo
echo "Conda environment: ${ENV_NAME}"
echo "MEGaNorm branch:   ${MEGANORM_BRANCH}"
echo "PCNtoolkit branch: ${PCNTOOLKIT_BRANCH}"
echo "Installation mode: ${INSTALL_MODE}"
echo

command -v git >/dev/null 2>&1 || {
    echo "ERROR: git is not available."
    exit 1
}

command -v conda >/dev/null 2>&1 || {
    echo "ERROR: conda is not available. Install Conda or Miniforge and try again."
    exit 1
}

check_repo() {
    local repo_dir="$1"
    local repo_name="$2"

    if ! git -C "${repo_dir}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        echo "ERROR: ${repo_name} was not found as a Git repository at:"
        echo "  ${repo_dir}"
        exit 1
    fi

    if ! git -C "${repo_dir}" remote get-url origin >/dev/null 2>&1; then
        echo "ERROR: ${repo_name} does not have a Git remote named 'origin'."
        exit 1
    fi
}

has_tracked_changes() {
    local repo_dir="$1"

    ! git -C "${repo_dir}" diff --quiet ||
        ! git -C "${repo_dir}" diff --cached --quiet
}

prepare_branch() {
    local repo_dir="$1"
    local branch="$2"
    local repo_name="$3"
    local require_remote_branch="$4"

    echo
    echo "Preparing ${repo_name} branch '${branch}'..."

    if has_tracked_changes "${repo_dir}"; then
        echo "ERROR: ${repo_name} has tracked uncommitted changes."
        echo "Commit or stash them before running the setup script."
        exit 1
    fi

    git -C "${repo_dir}" fetch origin --prune

    if git -C "${repo_dir}" show-ref --verify --quiet "refs/heads/${branch}"; then
        git -C "${repo_dir}" switch "${branch}"
    elif git -C "${repo_dir}" show-ref --verify --quiet "refs/remotes/origin/${branch}"; then
        git -C "${repo_dir}" switch --track -c "${branch}" "origin/${branch}"
    else
        echo "ERROR: Branch '${branch}' was not found for ${repo_name}."
        exit 1
    fi

    if git -C "${repo_dir}" show-ref --verify --quiet "refs/remotes/origin/${branch}"; then
        if ! git -C "${repo_dir}" pull --ff-only origin "${branch}"; then
            echo "ERROR: ${repo_name} branch '${branch}' could not be fast-forwarded."
            echo "Resolve the local/remote branch divergence manually and try again."
            exit 1
        fi
    elif [[ "${require_remote_branch}" == "true" ]]; then
        echo "ERROR: Required remote branch 'origin/${branch}' was not found for ${repo_name}."
        exit 1
    else
        echo "No remote branch 'origin/${branch}' was found."
        echo "Using the existing local ${repo_name} branch '${branch}'."
    fi

    echo "${repo_name} branch ready: ${branch}"
}

check_repo "${MEGANORM_DIR}" "MEGaNorm"

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

check_repo "${PCNTOOLKIT_DIR}" "PCNtoolkit"

prepare_branch "${MEGANORM_DIR}" "${MEGANORM_BRANCH}" "MEGaNorm" "false"
prepare_branch "${PCNTOOLKIT_DIR}" "${PCNTOOLKIT_BRANCH}" "PCNtoolkit" "true"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo
    echo "Conda environment '${ENV_NAME}' already exists."
    echo "Refreshing the existing environment..."

    CURRENT_PYTHON="$(
        conda run -n "${ENV_NAME}" \
            python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
    )"

    if [[ "${CURRENT_PYTHON}" != "${PYTHON_VERSION}" ]]; then
        echo "Updating Python from ${CURRENT_PYTHON} to ${PYTHON_VERSION}..."

        conda install \
            --channel=conda-forge \
            --strict-channel-priority \
            --name "${ENV_NAME}" \
            "python=${PYTHON_VERSION}" \
            --yes
    else
        echo "Python ${PYTHON_VERSION} requirement already satisfied."
    fi
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

if [[ "${INSTALL_MODE}" == "editable" ]]; then
    echo
    echo "Installing MEGaNorm in editable mode..."

    conda run -n "${ENV_NAME}" \
        python -m pip install -e "${MEGANORM_DIR}[dev]"

    echo
    echo "Installing PCNtoolkit in editable mode..."

    conda run -n "${ENV_NAME}" \
        python -m pip install -e "${PCNTOOLKIT_DIR}"
else
    echo
    echo "Installing MEGaNorm in shared mode..."

    conda run -n "${ENV_NAME}" \
        python -m pip install --force-reinstall "${MEGANORM_DIR}[dev]"

    echo
    echo "Installing PCNtoolkit in shared mode..."

    conda run -n "${ENV_NAME}" \
        python -m pip install --force-reinstall "${PCNTOOLKIT_DIR}"
fi

echo
echo "Verifying development environment..."

VERIFY_DIR="$(mktemp -d)"
trap 'rm -rf "${VERIFY_DIR}"' EXIT

(
    cd "${VERIFY_DIR}"

    conda run -n "${ENV_NAME}" env \
        INSTALL_MODE="${INSTALL_MODE}" \
        MEGANORM_EXPECTED="${MEGANORM_DIR}" \
        PCNTOOLKIT_EXPECTED="${PCNTOOLKIT_DIR}" \
        python - <<'PY'
import os
import sys
from pathlib import Path

import meganorm
import pcntoolkit

install_mode = os.environ["INSTALL_MODE"]

meganorm_path = Path(meganorm.__file__).resolve()
pcntoolkit_path = Path(pcntoolkit.__file__).resolve()

meganorm_expected = Path(os.environ["MEGANORM_EXPECTED"]).resolve()
pcntoolkit_expected = Path(os.environ["PCNTOOLKIT_EXPECTED"]).resolve()
environment_prefix = Path(sys.prefix).resolve()

print(f"Python:            {sys.version.split()[0]}")
print(f"Installation mode: {install_mode}")
print(f"MEGaNorm:          {meganorm_path}")
print(f"PCNtoolkit:        {pcntoolkit_path}")

if install_mode == "editable":
    if meganorm_expected not in meganorm_path.parents:
        raise RuntimeError(
            "MEGaNorm is not being imported from the local editable checkout."
        )

    if pcntoolkit_expected not in pcntoolkit_path.parents:
        raise RuntimeError(
            "PCNtoolkit is not being imported from the local editable checkout."
        )
else:
    if environment_prefix not in meganorm_path.parents:
        raise RuntimeError(
            "MEGaNorm is not being imported from the Conda environment."
        )

    if environment_prefix not in pcntoolkit_path.parents:
        raise RuntimeError(
            "PCNtoolkit is not being imported from the Conda environment."
        )
PY
)

echo
echo "======================================="
echo "MEGaNorm Development Environment is ready."

if [[ "${INSTALL_MODE}" == "shared" ]]; then
    echo
    echo "Shared mode is active."
    echo "Rerun this script after source updates to deploy the new code."
fi

echo
echo "Activate the environment with:"
echo
echo "  conda activate ${ENV_NAME}"
echo
