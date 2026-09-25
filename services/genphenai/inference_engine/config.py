from pathlib import Path

REGISTRY_VERSION = 1

DEFAULT_CHECKPOINTS_DIR = "checkpoints"
DEFAULT_ARTIFACTS_DIR = "artifacts"
DEFAULT_STATE_DIR = "state"
DEFAULT_REGISTRY_NAME = "checkpoint_registry.json"


def checkpoints_dir(bundle_dir: Path) -> Path:
    return bundle_dir / DEFAULT_CHECKPOINTS_DIR


def artifacts_dir(bundle_dir: Path) -> Path:
    return bundle_dir / DEFAULT_ARTIFACTS_DIR


def state_dir(bundle_dir: Path) -> Path:
    return bundle_dir / DEFAULT_STATE_DIR


def registry_path(bundle_dir: Path) -> Path:
    return state_dir(bundle_dir) / DEFAULT_REGISTRY_NAME
