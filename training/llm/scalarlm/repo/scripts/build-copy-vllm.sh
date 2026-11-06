#!/bin/bash
set -e

VLLM_SOURCE=$1
DEST_DIR=$2
LOCAL_PATH=$3
REPO_URL=$4
BRANCH=$5
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

    rm -rf $DEST_DIR/.git

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

if [ -n "${VLLM_FORK_PATCH:-}" ]; then
    SCRIPT_DIR_EARLY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    case "$VLLM_FORK_PATCH" in
        /*) FORK_PATCH="$VLLM_FORK_PATCH" ;;
        *)  FORK_PATCH="${SCRIPT_DIR_EARLY}/${VLLM_FORK_PATCH}" ;;
    esac
    if [ ! -f "$FORK_PATCH" ]; then
        echo "❌ Error: VLLM_FORK_PATCH=$VLLM_FORK_PATCH not found at $FORK_PATCH"
        exit 1
    fi
    echo "🩹 Applying vLLM fork patch $FORK_PATCH"
    git -C "$DEST_DIR" apply --whitespace=nowarn "$FORK_PATCH"
fi

echo "📍 vLLM is ready at: $DEST_DIR"
ls -la "$DEST_DIR" | head -5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHER="${SCRIPT_DIR}/apply_patches.py"
if [ -f "${PATCHER}" ] && [ -r "${PATCHER}" ]; then
    echo "🩹 Applying ScalarLM vLLM-fork patches"
    python3 "${PATCHER}" "${DEST_DIR}"
else
    echo "⚠️  Patcher not found at ${PATCHER}; skipping"
fi
