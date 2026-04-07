#!/usr/bin/env bash

set -euo pipefail

usage() {
  echo "Usage: ./deploy-gh-pages.sh <file-to-deploy> [commit-message]"
  echo "Example: ./deploy-gh-pages.sh alchemy_v1.html"
  echo "Example: ./deploy-gh-pages.sh globe/globe_v1.html \"Deploy globe demo\""
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

TARGET_FILE="$1"
COMMIT_MESSAGE="${2:-Deploy ${TARGET_FILE} to GitHub Pages}"

if [[ ! -f "$TARGET_FILE" ]]; then
  echo "Error: file not found: $TARGET_FILE"
  exit 1
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: this script must be run inside a git repository."
  exit 1
fi

if ! git remote get-url origin >/dev/null 2>&1; then
  echo "Error: git remote 'origin' is not configured."
  exit 1
fi

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Error: please commit or stash your changes before deploying."
  exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

cp "$TARGET_FILE" "$TMP_DIR/index.html"
touch "$TMP_DIR/.nojekyll"

if [[ -f "$REPO_ROOT/CNAME" ]]; then
  cp "$REPO_ROOT/CNAME" "$TMP_DIR/CNAME"
fi

pushd "$TMP_DIR" >/dev/null
git init >/dev/null
git checkout -b gh-pages >/dev/null
git add . >/dev/null
git commit -m "$COMMIT_MESSAGE" >/dev/null
git remote add origin "$(git -C "$REPO_ROOT" remote get-url origin)"
git push -f origin gh-pages >/dev/null
popd >/dev/null

echo "Deployed '$TARGET_FILE' to branch 'gh-pages'."
echo "If Pages is enabled for the gh-pages branch, your site will update shortly."
