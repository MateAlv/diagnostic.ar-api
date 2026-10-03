#!/usr/bin/env bash
# Downloads the bundles pinned in model.lock (the model and, when pinned, its
# re-ranker) and checks their sha256.
#
# Leaves the tarball in model-dist/ (what the Dockerfile copies) and unpacks
# it into model/ (what the tests and a local uvicorn load). Uses GITHUB_TOKEN
# when set, as in CI; otherwise the gh CLI's own login.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=/dev/null
. "$here/model.lock"

dist="$here/model-dist"
mkdir -p "$dist" "$here/model"

# fetch ASSET SHA256 TAG: download unless the copy on disk already matches.
fetch() {
    local asset=$1 sha=$2 tag=$3 tarball="$dist/$1"
    if ! echo "$sha  $tarball" | sha256sum -c --status 2>/dev/null; then
        rm -f "$tarball"
        GH_TOKEN=${GITHUB_TOKEN:-${GH_TOKEN:-}} gh release download "$tag" \
            --repo "$MODEL_REPO" --pattern "$asset" --dir "$dist"
        echo "$sha  $tarball" | sha256sum -c --quiet
    fi
    rm -rf "$here/model/${asset%.tar.gz}"
    tar -xzf "$tarball" -C "$here/model"
    echo "$tag ready: $here/model/${asset%.tar.gz}"
}

fetch "$MODEL_ASSET" "$MODEL_SHA256" "$MODEL_TAG"
if [ -n "${RERANK_TAG:-}" ]; then
    fetch "$RERANK_ASSET" "$RERANK_SHA256" "$RERANK_TAG"
fi
