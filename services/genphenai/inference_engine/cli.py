from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference_engine.checkpoint_registry import CheckpointRegistryManager
from inference_engine.engine import GenpheniaEngine
from inference_engine.types import CheckpointSpec, normalize_model_version


def _build_spec_from_args(args: argparse.Namespace) -> CheckpointSpec:
    model_version = normalize_model_version(args.model_version)
    num_heads = int(args.num_heads)
    if model_version != "2.5":
        num_heads = 4
    spec = CheckpointSpec(
        checkpoint_id=args.checkpoint_id,
        checkpoint_path=args.checkpoint_path,
        model_version=model_version,
        in_channels=int(args.in_channels),
        hidden_channels=int(args.hidden_channels),
        num_heads=num_heads,
        genes_path=args.genes_path,
        hp_json_path=args.hp_json_path,
        embeddings_data_path=args.embeddings_data_path,
        embeddings_index_path=args.embeddings_index_path,
        onnx_path=args.onnx_path,
        include_ancestors=bool(args.include_ancestors),
        include_reverse_edges=bool(args.include_reverse_edges),
        checkpoint_key=args.checkpoint_key,
    )
    spec.validate()
    return spec


def _cmd_init(args: argparse.Namespace) -> None:
    manager = CheckpointRegistryManager(Path(args.bundle))
    manager.save_state()
    print(
        json.dumps(
            {
                "bundle": str(manager.bundle_dir),
                "checkpoints_dir": str(manager.checkpoints_dir),
                "artifacts_dir": str(manager.artifacts_dir),
                "registry_path": str(manager.registry_path),
            },
            indent=2,
        )
    )


def _cmd_create_spec(args: argparse.Namespace) -> None:
    manager = CheckpointRegistryManager(Path(args.bundle))
    spec = _build_spec_from_args(args)
    spec_path = manager.install_checkpoint_package(
        spec,
        copy_checkpoint=bool(args.copy_checkpoint),
    )
    print(json.dumps({"spec_path": str(spec_path)}, indent=2))


def _cmd_register_checkpoint(args: argparse.Namespace) -> None:
    manager = CheckpointRegistryManager(Path(args.bundle))
    spec = _build_spec_from_args(args)
    spec_path = manager.install_checkpoint_package(
        spec,
        copy_checkpoint=bool(args.copy_checkpoint),
    )
    record = manager.register_spec_file(spec_path, activate=bool(args.activate))
    print(json.dumps(record.to_dict(), indent=2))


def _cmd_sync_checkpoints(args: argparse.Namespace) -> None:
    manager = CheckpointRegistryManager(Path(args.bundle))
    changed = manager.sync_checkpoint_packages(
        auto_activate_latest=bool(args.auto_activate_latest)
    )
    active = None
    try:
        active = manager.get_active_record().checkpoint_id
    except Exception:
        pass
    print(json.dumps({"changed": changed, "active_checkpoint_id": active}, indent=2))


def _cmd_list_checkpoints(args: argparse.Namespace) -> None:
    manager = CheckpointRegistryManager(Path(args.bundle))
    payload = {
        "active_checkpoint_id": manager.state.active_checkpoint_id,
        "checkpoints": [record.to_dict() for record in manager.list_records()],
    }
    print(json.dumps(payload, indent=2))


def _cmd_set_active(args: argparse.Namespace) -> None:
    manager = CheckpointRegistryManager(Path(args.bundle))
    manager.set_active(args.checkpoint_id)
    print(json.dumps({"active_checkpoint_id": args.checkpoint_id}, indent=2))


def _cmd_predict(args: argparse.Namespace) -> None:
    engine = GenpheniaEngine.load(
        args.bundle,
        auto_refresh_seconds=float(args.auto_refresh_seconds),
        auto_activate_latest=bool(args.auto_activate_latest),
    )
    result = engine.predict(
        hpo_ids=list(args.hpo_ids),
        top_k=int(args.top_k),
        return_distribution=bool(args.return_distribution),
    )
    print(json.dumps(result, indent=2))


def _cmd_health(args: argparse.Namespace) -> None:
    engine = GenpheniaEngine.load(
        args.bundle,
        auto_refresh_seconds=float(args.auto_refresh_seconds),
        auto_activate_latest=bool(args.auto_activate_latest),
    )
    print(json.dumps(engine.health(), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GenPhenia production inference engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize bundle layout")
    init_parser.add_argument("--bundle", type=str, default=".")
    init_parser.set_defaults(func=_cmd_init)

    spec_parent = argparse.ArgumentParser(add_help=False)
    spec_parent.add_argument("--bundle", type=str, default=".")
    spec_parent.add_argument("--checkpoint-id", type=str, required=True)
    spec_parent.add_argument("--checkpoint-path", type=str, required=True)
    spec_parent.add_argument(
        "--model-version",
        type=str,
        choices=["2.0", "2.5", "3.0", "v2.0", "v2.5", "v3.0"],
        required=True,
    )
    spec_parent.add_argument("--in-channels", type=int, default=768)
    spec_parent.add_argument("--hidden-channels", type=int, required=True)
    spec_parent.add_argument("--num-heads", type=int, default=4)
    spec_parent.add_argument("--genes-path", type=str, required=True)
    spec_parent.add_argument("--hp-json-path", type=str, required=True)
    spec_parent.add_argument("--embeddings-data-path", type=str, required=True)
    spec_parent.add_argument("--embeddings-index-path", type=str, required=True)
    spec_parent.add_argument("--onnx-path", type=str, default=None)
    spec_parent.add_argument("--include-ancestors", action=argparse.BooleanOptionalAction, default=True)
    spec_parent.add_argument("--include-reverse-edges", action=argparse.BooleanOptionalAction, default=False)
    spec_parent.add_argument("--checkpoint-key", type=str, default="model_state_dict")
    spec_parent.add_argument("--copy-checkpoint", action=argparse.BooleanOptionalAction, default=True)

    create_spec_parser = subparsers.add_parser(
        "create-spec",
        parents=[spec_parent],
        help="Create typed checkpoint package spec in checkpoints/<id>/spec.json",
    )
    create_spec_parser.set_defaults(func=_cmd_create_spec)

    register_parser = subparsers.add_parser(
        "register-checkpoint",
        parents=[spec_parent],
        help="Install a checkpoint package and build ONNX artifacts",
    )
    register_parser.add_argument("--activate", action=argparse.BooleanOptionalAction, default=True)
    register_parser.set_defaults(func=_cmd_register_checkpoint)

    sync_parser = subparsers.add_parser(
        "sync-checkpoints",
        help="Sync checkpoints/*/spec.json into registry and artifacts",
    )
    sync_parser.add_argument("--bundle", type=str, default=".")
    sync_parser.add_argument("--auto-activate-latest", action=argparse.BooleanOptionalAction, default=False)
    sync_parser.set_defaults(func=_cmd_sync_checkpoints)

    list_parser = subparsers.add_parser("list-checkpoints", help="List registered checkpoints")
    list_parser.add_argument("--bundle", type=str, default=".")
    list_parser.set_defaults(func=_cmd_list_checkpoints)

    set_active_parser = subparsers.add_parser("set-active", help="Set active checkpoint")
    set_active_parser.add_argument("--bundle", type=str, default=".")
    set_active_parser.add_argument("--checkpoint-id", type=str, required=True)
    set_active_parser.set_defaults(func=_cmd_set_active)

    predict_parser = subparsers.add_parser("predict", help="Run inference")
    predict_parser.add_argument("--bundle", type=str, default=".")
    predict_parser.add_argument("--hpo_ids", nargs="+", required=True)
    predict_parser.add_argument("--top_k", type=int, default=60)
    predict_parser.add_argument("--return-distribution", action=argparse.BooleanOptionalAction, default=False)
    predict_parser.add_argument("--auto-refresh-seconds", type=float, default=5.0)
    predict_parser.add_argument("--auto-activate-latest", action=argparse.BooleanOptionalAction, default=True)
    predict_parser.set_defaults(func=_cmd_predict)

    health_parser = subparsers.add_parser("health", help="Runtime health info")
    health_parser.add_argument("--bundle", type=str, default=".")
    health_parser.add_argument("--auto-refresh-seconds", type=float, default=5.0)
    health_parser.add_argument("--auto-activate-latest", action=argparse.BooleanOptionalAction, default=True)
    health_parser.set_defaults(func=_cmd_health)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
