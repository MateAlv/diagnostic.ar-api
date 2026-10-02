#!/usr/bin/env bash
# Downloads the HPO embeddings pinned in embeddings.lock, with the sha256 the
# release publishes beside them, and checks one against the other.
#
# Leaves both in embeddings-dist/ at the repository root, where
# Dockerfile.recognizer reads them through a bind mount (and checks them
# again). Not under services/, which the image copies whole: the package would
# end up in it twice. Always downloads afresh, so a stale local copy can never
# stand in for the pinned release.
# Uses GITHUB_TOKEN when set, as in CI; otherwise the gh CLI's own login.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
# shellcheck source=/dev/null
. "$here/embeddings.lock"

dist="$root/embeddings-dist"
rm -rf "$dist"
mkdir -p "$dist"

if ! GH_TOKEN=${GITHUB_TOKEN:-${GH_TOKEN:-}} gh release download "$EMB_TAG" \
        --repo "$EMB_REPO" --pattern "$EMB_ASSET" --pattern "$EMB_ASSET.sha256" --dir "$dist"; then
    echo "ERROR: could not download $EMB_TAG from $EMB_REPO." >&2
    echo "Is it published? It is published when hpo-embeddings/embeddings.lock changes on its main." >&2
    exit 1
fi
(cd "$dist" && sha256sum -c --quiet "$EMB_ASSET.sha256")
echo "$EMB_TAG ready: $dist/$EMB_ASSET ($(cut -d' ' -f1 "$dist/$EMB_ASSET.sha256"))"
