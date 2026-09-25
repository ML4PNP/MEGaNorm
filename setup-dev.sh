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
# MEGaNorm uses the branch supplied as the first argument. If no branch is
# supplied, "dev" is used.
#
# PCNtoolkit always uses the "integration" branch.
#
# The script:
#   - fetches and switches to the required Git branches,
#   - updates those branches with fast-forward-only pulls,
#   - creates or refreshes the meganorm-dev Conda environment,
#   - installs MEGaNorm and PCNtoolkit in editable mode,
#   - verifies that the local development checkouts are being imported.
#
# To protect local work, the script stops if tracked uncommitted changes are
# present before a branch switch or update.

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

Set up or refresh the MEGaNorm Development Environment.

Arguments:
  MEGANORM_BRANCH   MEGaNorm branch to use for development.
                    Optional. Defaults to "dev".

Expected directory layout:

  parent/
    MEGaNorm/
      setup-dev.sh
    pcntkdev/
      PCNtoolkit/

Examples:
  ./setup-dev.sh
  ./setup-dev.sh dev
  ./setup-dev.sh dev_meg
  ./setup-dev.sh feature/new-workflow

Behavior:
  - MEGaNorm is switched to the requested branch.
  - PCNtoolkit is switched to "integration".
  - Remote branches are fetched and updated using fast-forward-only pulls.
  - Existing tracked uncommitted changes cause the script to stop.
  - The "meganorm-dev" Conda environment is created if missing.
  - If it already exists, it is refreshed rather than recreated.
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
echo "MEGaNorm Development Environment setup"
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
    echo "Refreshing the existing development environment..."

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

echo
echo "Installing or refreshing MEGaNorm development dependencies..."

conda run -n "${ENV_NAME}" \
    python -m pip install -e "${MEGANORM_DIR}[dev]"

echo
echo "Installing or refreshing PCNtoolkit from the local integration branch..."

# Install PCNtoolkit last so the final environment always points to the local
# integration checkout rather than a released PCNtoolkit dependency.
conda run -n "${ENV_NAME}" \
    python -m pip install -e "${PCNTOOLKIT_DIR}"

echo
echo "Verifying development environment..."

conda run -n "${ENV_NAME}" env \
    MEGANORM_EXPECTED="${MEGANORM_DIR}" \
    PCNTOOLKIT_EXPECTED="${PCNTOOLKIT_DIR}" \
    python - <<'PY'
import os
import sys
from pathlib import Path

import meganorm
import pcntoolkit

meganorm_path = Path(meganorm.__file__).resolve()
pcntoolkit_path = Path(pcntoolkit.__file__).resolve()

meganorm_expected = Path(os.environ["MEGANORM_EXPECTED"]).resolve()
pcntoolkit_expected = Path(os.environ["PCNTOOLKIT_EXPECTED"]).resolve()

print(f"Python:     {sys.version.split()[0]}")
print(f"MEGaNorm:   {meganorm_path}")
print(f"PCNtoolkit: {pcntoolkit_path}")

if meganorm_expected not in meganorm_path.parents:
    raise RuntimeError(
        "MEGaNorm is not being imported from the local development checkout."
    )

if pcntoolkit_expected not in pcntoolkit_path.parents:
    raise RuntimeError(
        "PCNtoolkit is not being imported from the local development checkout."
    )
PY

echo
echo "======================================="
echo "MEGaNorm Development Environment is ready."
echo
echo "Activate the environment with:"
echo
echo "  conda activate ${ENV_NAME}"
echo
