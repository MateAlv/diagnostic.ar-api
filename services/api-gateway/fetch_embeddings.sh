#!/usr/bin/env bash
# Downloads the HPO embeddings pinned in embeddings.lock and checks their sha256.
#
# Leaves the package in embeddings-dist/ at the repository root, where
# Dockerfile.recognizer reads it through a bind mount. Not under services/,
# which the image copies whole: the package would end up in it twice.
# Uses GITHUB_TOKEN when set, as in CI; otherwise the gh CLI's own login.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
# shellcheck source=/dev/null
. "$here/embeddings.lock"

if ! [[ "$EMB_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "ERROR: EMB_SHA256 in embeddings.lock is '$EMB_SHA256'; set it to the sha256 of $EMB_TAG" >&2
    exit 1
fi

dist="$root/embeddings-dist"
package="$dist/$EMB_ASSET"
mkdir -p "$dist"

if ! echo "$EMB_SHA256  $package" | sha256sum -c --status 2>/dev/null; then
    rm -f "$package"
    GH_TOKEN=${GITHUB_TOKEN:-${GH_TOKEN:-}} gh release download "$EMB_TAG" \
        --repo "$EMB_REPO" --pattern "$EMB_ASSET" --dir "$dist"
    echo "$EMB_SHA256  $package" | sha256sum -c --quiet
fi
echo "$EMB_TAG ready: $package"
