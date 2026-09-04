#!/bin/bash
# Script used by Docker build process to handle vLLM source:
#   - either copy local or clone remote
set -e

VLLM_SOURCE=$1
DEST_DIR=$2
LOCAL_PATH=$3
REPO_URL=$4
BRANCH=$5
# Optional pinned commit. When set, we `git checkout` it after the
# clone so builds are reproducible across time even if BRANCH advances.
# Empty = use BRANCH tip (matches the pre-pin behavior).
COMMIT=$6

echo "🔧 Setting up vLLM source..."
echo "   Source type: $VLLM_SOURCE"

if [ "$VLLM_SOURCE" = "local" ]; then
    echo "📁 Using local vLLM from: $LOCAL_PATH"

    if [ ! -d "$LOCAL_PATH" ]; then
        echo "❌ Error: Local vLLM directory not found at $LOCAL_PATH"
        echo ""
        echo "   For local development, vLLM must be cloned into the ScalarLM directory:"
        echo "   cd /path/to/scalarlm"
        echo "   git clone -b main https://github.com/supermassive-intelligence/vllm-fork.git vllm"
        echo ""
        echo "   This will create: scalarlm/vllm/"
        exit 1
    fi

    echo "📋 Copying local vLLM to $DEST_DIR..."
    cp -r "$LOCAL_PATH" "$DEST_DIR"

    # Copy only essential git metadata for version detection (not objects)
    rm -rf $DEST_DIR/.git

    # Create a minimal but valid git repository
    cd "$DEST_DIR"
    git init
    git config user.name "Docker Build"
    git config user.email "build@docker.local"
    git add -A
    git commit -m "Build snapshot" --no-verify
    git tag -a v0.6.5 -m "Version 0.6.5"
    cd -
    echo "📌 Keeping git metadata for version detection"

    echo "✅ Local vLLM copied successfully"

else
    echo "🌐 Cloning vLLM from remote repository"
    echo "   Repository: $REPO_URL"
    echo "   Branch: $BRANCH"

    git clone -b "$BRANCH" "$REPO_URL" "$DEST_DIR"

    if [ -n "$COMMIT" ]; then
        echo "📌 Pinning to commit: $COMMIT"
        git -C "$DEST_DIR" checkout "$COMMIT"
    fi

    echo "✅ Remote vLLM cloned successfully"
fi

echo "📍 vLLM is ready at: $DEST_DIR"
ls -la "$DEST_DIR" | head -5

# ScalarLM fork patches (see scripts/vllm_patches/apply_patches.py for why
# each one exists and what anchors it's guarding). Runs after the vLLM
# tree is staged, before the pip install compiles anything. In-container
# the patcher lives next to this script (single COPY; see Dockerfile),
# so source it from SCRIPT_DIR directly rather than a vllm_patches/ subdir.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHER="${SCRIPT_DIR}/apply_patches.py"
# We invoke the patcher via `python3 ${PATCHER}`, so it doesn't need the
# exec bit — check for readability instead so a file missing +x (e.g.
# after some cp/rsync that didn't preserve mode) doesn't silently skip
# patch application.
if [ -f "${PATCHER}" ] && [ -r "${PATCHER}" ]; then
    echo "🩹 Applying ScalarLM vLLM-fork patches"
    python3 "${PATCHER}" "${DEST_DIR}"
else
    echo "⚠️  Patcher not found at ${PATCHER}; skipping"
fi
