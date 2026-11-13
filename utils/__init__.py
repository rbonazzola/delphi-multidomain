"""
Public API for the utils package.

Import from here rather than from individual submodules or from utils.utils.

Submodules are imported lazily (PEP 562), so that e.g. utils.ckpt_utils can be used
to load a checkpoint in an environment without mlflow installed.
"""

_SUBMODULE_OF = {
    # mlflow_utils
    "setup_mlflow": "utils.mlflow_utils",
    "load_run_params": "utils.mlflow_utils",
    "load_checkpoint": "utils.mlflow_utils",
    "get_checkpoint_path": "utils.mlflow_utils",
    "parse_domains_param": "utils.mlflow_utils",
    "RunSetup": "utils.mlflow_utils",
    "get_run_setup": "utils.mlflow_utils",
    "get_run_setup_for_run": "utils.mlflow_utils",
    # ckpt_utils
    "strip_compiled_prefix": "utils.ckpt_utils",
    # run_loader
    "reconstruct_from_run": "utils.run_loader",
    "reconstruct_model": "utils.run_loader",
    "config_from_runid": "utils.run_loader",
    "AUTO_BLOCK_SIZE": "utils.run_loader",
    # utils
    "load_domain_config": "utils.utils",
    "apply_domain_overrides": "utils.utils",
    "read_ids": "utils.utils",
    "read_disease_ids": "utils.utils",
}

__all__ = list(_SUBMODULE_OF)


def __getattr__(name):
    if name not in _SUBMODULE_OF:
        raise AttributeError(f"module 'utils' has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(_SUBMODULE_OF[name]), name)
