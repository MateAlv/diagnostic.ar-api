#!/usr/bin/env bash
# Downloads the bundle pinned in model.lock and checks its sha256.
#
# Leaves the tarball in model-dist/ (what the Dockerfile copies) and unpacks
# it into model/ (what the tests and a local uvicorn load). Uses GITHUB_TOKEN
# when set, as in CI; otherwise the gh CLI's own login.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=/dev/null
. "$here/model.lock"

dist="$here/model-dist"
tarball="$dist/$MODEL_ASSET"
mkdir -p "$dist"

if ! echo "$MODEL_SHA256  $tarball" | sha256sum -c --status 2>/dev/null; then
    rm -f "$tarball"
    GH_TOKEN=${GITHUB_TOKEN:-${GH_TOKEN:-}} gh release download "$MODEL_TAG" \
        --repo "$MODEL_REPO" --pattern "$MODEL_ASSET" --dir "$dist"
    echo "$MODEL_SHA256  $tarball" | sha256sum -c --quiet
fi

bundle="$here/model/${MODEL_ASSET%.tar.gz}"
rm -rf "$bundle"
tar -xzf "$tarball" -C "$here/model"
echo "$MODEL_TAG ready: $bundle"
