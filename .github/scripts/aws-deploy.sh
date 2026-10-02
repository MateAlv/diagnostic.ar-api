#!/usr/bin/env bash
# Deploys freshly pushed images to the AWS copy of diagnostic.ar, from a GitHub
# Actions run that already holds the deploy role's credentials
# (aws-actions/configure-aws-credentials). The same file lives in
# diagnostic.ar, diagnostic.ar-api and spanish-symptoms-recognizer; keep the
# three identical. How it fits together: docs/AWS_CICD.md in diagnostic.ar.
#
#   aws-deploy.sh deploy BACKEND_TAG=app-1a2b3c4 FRONTEND_TAG=app-1a2b3c4
#   aws-deploy.sh check  web|model|recognizer ...
#   aws-deploy.sh rollback          (puts back the tags the last deploy replaced)
#
# The instance only ever runs the fixed `diagnosticar deploy` document; this
# script cannot run anything else there, and it never sees a secret.
set -euo pipefail

ci() { aws ssm get-parameter --name "/diagnosticar-ci/$1" --query Parameter.Value --output text; }
log() { printf '==> %s\n' "$*"; }
die() { printf '::error::%s\n' "$*" >&2; exit 1; }

STATE="${RUNNER_TEMP:-/tmp}/aws-deploy-previous"

run_deploy() {
    local instance doc id status
    instance=$(ci INSTANCE_ID)
    doc=$(ci DEPLOY_DOC)
    id=$(aws ssm send-command --instance-ids "$instance" --document-name "$doc" \
            --comment "${GITHUB_REPOSITORY:-manual}@${GITHUB_SHA:0:7}" \
            --query Command.CommandId --output text)
    log "Deploying on $instance (command $id)"
    for _ in $(seq 1 180); do
        status=$(aws ssm get-command-invocation --command-id "$id" --instance-id "$instance" \
                    --query Status --output text 2>/dev/null || echo Pending)
        case "$status" in Pending|InProgress|Delayed) sleep 5 ;; *) break ;; esac
    done
    aws ssm get-command-invocation --command-id "$id" --instance-id "$instance" \
        --query StandardOutputContent --output text
    [ "$status" = "Success" ] || die "deploy on the instance ended $status"
}

cmd_deploy() {
    [ $# -gt 0 ] || die "usage: aws-deploy.sh deploy NAME=TAG ..."
    : > "$STATE"
    local kv name tag old
    for kv in "$@"; do
        name=${kv%%=*}; tag=${kv#*=}
        [[ "$name" =~ ^(BACKEND|FRONTEND|RECOGNIZER|MODEL)_TAG$ ]] || die "unexpected tag name $name"
        old=$(aws ssm get-parameter --name "/diagnosticar/config/$name" \
                --query Parameter.Value --output text 2>/dev/null || true)
        echo "$name=$old" >> "$STATE"
        aws ssm put-parameter --name "/diagnosticar/config/$name" --type String \
            --value "$tag" --overwrite >/dev/null
        log "$name: ${old:-(IMAGE_TAG)} -> $tag"
    done
    run_deploy
}

cmd_rollback() {
    [ -s "$STATE" ] || die "nothing to roll back"
    local name old
    while IFS='=' read -r name old; do
        if [ -n "$old" ]; then
            aws ssm put-parameter --name "/diagnosticar/config/$name" --type String \
                --value "$old" --overwrite >/dev/null
        else
            aws ssm delete-parameter --name "/diagnosticar/config/$name" >/dev/null
        fi
        log "$name back to ${old:-(IMAGE_TAG)}"
    done < "$STATE"
    run_deploy
}

# Waits up to two minutes for each service to answer healthy through the
# public URL: containers need a few seconds (migrations, the encoder) after
# the deploy returns.
wait_for() {  # wait_for NAME URL PYTHON-CHECK
    local body
    for _ in $(seq 1 24); do
        body=$(curl -s --max-time 10 "$2" || true)
        if printf '%s' "$body" | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if ($3) else 1)" 2>/dev/null; then
            log "$1 healthy"
            return 0
        fi
        sleep 5
    done
    die "$1 not healthy at $2: ${body:0:300}"
}

cmd_check() {
    local url what
    url=$(ci PUBLIC_URL)
    for what in "$@"; do
        case "$what" in
            web)
                wait_for backend "$url/api/health" 'd.get("status") == "ok"' ;;
            model)
                wait_for model "$url/genphenai/healthz" \
                    'd.get("status") == "ok" and d.get("mode") == "in-process" and d.get("specialties_available")'
                # A ranking, end to end: the reference case of docs/CLOUD_DEPLOY.md.
                curl -sf --max-time 20 -X POST "$url/genphenai/rank-genes" -H 'Content-Type: application/json' \
                    -d '{"hpo_ids":["HP:0000957","HP:0001067","HP:0009737","HP:0000316"],"top_k":3}' \
                | python3 -c 'import json,sys; r=[g["gene_symbol"] for g in json.load(sys.stdin)["results"]]; print("    ranking:", r); sys.exit(0 if r else 1)' \
                    || die "rank-genes did not answer a ranking" ;;
            recognizer)
                # ready may be false until the Anthropic key is set; the
                # service must still be up, with its HPO index loaded.
                wait_for recognizer "$url/hpo-api/healthz" \
                    'd.get("status") == "ok" and d.get("hpo_index_loaded")' ;;
            *) die "unknown check $what" ;;
        esac
    done
}

case "${1:-}" in
    deploy)   shift; cmd_deploy "$@" ;;
    rollback) cmd_rollback ;;
    check)    shift; cmd_check "$@" ;;
    *)        sed -n '2,14p' "$0"; exit 1 ;;
esac
