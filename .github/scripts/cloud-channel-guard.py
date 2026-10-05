#!/usr/bin/env python3
"""Decides whether this run may move the extractor's `cloud` channel.

The extractor image (diagnosticar-cloud-hpo-recognizer) is built from two
repositories, MateAlv/diagnostic.ar-api and MateAlv/spanish-symptoms-recognizer,
and both publish it: each takes its own commit and the other's latest main.
Without this check the channel ends on whichever run finishes last, which can
be the older code: the API merges, its run takes recognizer A; the recognizer
merges B; the recognizer's run finishes first with B; then the API's finishes
and puts A back.

Every extractor image carries both commits as labels. Before moving the
channel, this compares the image it points to with the one just built, in
each repository (GitHub's compare API):

  not behind in either        -> move     (the new image has everything the channel has)
  behind in one, ahead in none -> skip     (the channel is already newer)
  behind in one, ahead in other -> rebuild (each has a commit the other lacks:
                                            a new run takes both latest mains)

A channel image without the labels (built before them) is replaced.

The same file lives in both repositories; keep the two identical.

Usage:
  cloud-channel-guard.py IMAGE API_COMMIT RECOGNIZER_COMMIT
      IMAGE is the repository without a tag, e.g.
      ghcr.io/matealv/diagnosticar-cloud-hpo-recognizer
  cloud-channel-guard.py --self-test
Prints one word on stdout, move | skip | rebuild, and why on stderr.

Environment:
  REGISTRY_USER, REGISTRY_TOKEN  read the channel's image from the registry
  GH_TOKEN                       read both repositories (contents: read)
"""
import base64
import json
import os
import sys
import urllib.error
import urllib.request

CHANNEL = "cloud"
LABELS = {
    "api": "ar.diagnostic.api.commit",
    "recognizer": "ar.diagnostic.recognizer.commit",
}
REPOS = {
    "api": "MateAlv/diagnostic.ar-api",
    "recognizer": "MateAlv/spanish-symptoms-recognizer",
}

INDEX_TYPES = (
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
)
MANIFEST_TYPES = (
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
)


def log(msg):
    print(msg, file=sys.stderr)


# --- The decision ---------------------------------------------------------------

def decide(status):
    """status: {"api": s, "recognizer": s}, each as GitHub's compare API says
    how the new commit relates to the channel's: identical, ahead, behind or
    diverged. Diverged (a rewritten main) counts as behind: never replace what
    cannot be shown to be older."""
    behind = [k for k, s in status.items() if s in ("behind", "diverged")]
    ahead = [k for k, s in status.items() if s == "ahead"]
    if not behind:
        return "move"
    if ahead:
        return "rebuild"
    return "skip"


# --- The registry -------------------------------------------------------------

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def _get(url, headers, follow=True):
    opener = urllib.request.build_opener() if follow else urllib.request.build_opener(_NoRedirect)
    req = urllib.request.Request(url, headers=headers)
    try:
        with opener.open(req, timeout=30) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def channel_labels(image):
    """The labels of IMAGE:cloud for linux/amd64, or None if there is no channel."""
    host, path = image.split("/", 1)
    basic = base64.b64encode(f"{os.environ['REGISTRY_USER']}:{os.environ['REGISTRY_TOKEN']}".encode()).decode()
    code, _, body = _get(
        f"https://{host}/token?service={host}&scope=repository:{path}:pull",
        {"Authorization": f"Basic {basic}"},
    )
    if code != 200:
        raise SystemExit(f"registry login failed (HTTP {code})")
    auth = {"Authorization": f"Bearer {json.loads(body)['token']}"}

    def manifest(ref):
        code, _, body = _get(
            f"https://{host}/v2/{path}/manifests/{ref}",
            {**auth, "Accept": ", ".join(INDEX_TYPES + MANIFEST_TYPES)},
        )
        if code == 404:
            return None
        if code != 200:
            raise SystemExit(f"reading {image}:{ref} failed (HTTP {code})")
        return json.loads(body)

    m = manifest(CHANNEL)
    if m is None:
        return None
    if m.get("mediaType") in INDEX_TYPES or "manifests" in m:
        amd64 = [d for d in m["manifests"]
                 if d.get("platform", {}).get("os") == "linux"
                 and d.get("platform", {}).get("architecture") == "amd64"]
        if not amd64:
            raise SystemExit(f"{image}:{CHANNEL} has no linux/amd64 image")
        m = manifest(amd64[0]["digest"])

    # The config blob redirects to a CDN, which must not receive the token.
    url = f"https://{host}/v2/{path}/blobs/{m['config']['digest']}"
    code, headers, body = _get(url, auth, follow=False)
    if code in (301, 302, 303, 307, 308):
        code, _, body = _get(headers["Location"], {})
    if code != 200:
        raise SystemExit(f"reading the image config failed (HTTP {code})")
    return json.loads(body).get("config", {}).get("Labels") or {}


# --- GitHub --------------------------------------------------------------------

def compare(repo, base, head):
    """How head relates to base in repo: identical, ahead, behind or diverged."""
    if base == head:
        return "identical"
    code, _, body = _get(
        f"https://api.github.com/repos/{repo}/compare/{base}...{head}",
        {"Authorization": f"Bearer {os.environ['GH_TOKEN']}",
         "Accept": "application/vnd.github+json"},
    )
    if code != 200:
        raise SystemExit(f"comparing {repo} {base[:7]}...{head[:7]} failed (HTTP {code})")
    return json.loads(body)["status"]


# --- Main ----------------------------------------------------------------------

def self_test():
    cases = {
        ("identical", "identical"): "move",
        ("ahead", "identical"): "move",
        ("identical", "ahead"): "move",
        ("ahead", "ahead"): "move",
        ("identical", "behind"): "skip",     # the API finished last with an older recognizer
        ("behind", "identical"): "skip",     # the recognizer finished last with an older API
        ("behind", "behind"): "skip",
        ("ahead", "behind"): "rebuild",      # each has a commit the other lacks
        ("behind", "ahead"): "rebuild",
        ("diverged", "identical"): "skip",
        ("diverged", "ahead"): "rebuild",
    }
    for (api, rec), want in cases.items():
        got = decide({"api": api, "recognizer": rec})
        assert got == want, f"api={api} recognizer={rec}: {got}, want {want}"
    print(f"ok: {len(cases)} cases")


def main(argv):
    if argv[1:] == ["--self-test"]:
        self_test()
        return
    if len(argv) != 4:
        raise SystemExit(__doc__)
    image, ours = argv[1], {"api": argv[2], "recognizer": argv[3]}

    labels = channel_labels(image)
    if labels is None:
        log(f"{image}:{CHANNEL} does not exist yet")
        print("move")
        return
    theirs = {k: labels.get(label) for k, label in LABELS.items()}
    if not all(theirs.values()):
        log(f"{image}:{CHANNEL} predates the commit labels; replacing it")
        print("move")
        return

    status = {k: compare(REPOS[k], theirs[k], ours[k]) for k in REPOS}
    for k in REPOS:
        log(f"  {k:10} channel {theirs[k][:7]}  this run {ours[k][:7]}  -> {status[k]}")
    print(decide(status))


if __name__ == "__main__":
    main(sys.argv)
