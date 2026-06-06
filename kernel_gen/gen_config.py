"""Shared helpers for printing a run summary and persisting the generation config.

Used by the generate_kernels_*/generate_kernelbook_* scripts so every run prints a
short summary up front and drops a ``generation_config.yaml`` into its output dir
recording the full configuration (model, dataset, sampling params, paths, …).
"""

import os


def print_generation_summary(config: dict, keys=None, title="Generation configuration"):
    """Print a compact summary of the run.

    ``keys`` selects (and orders) which fields to show; defaults to every field
    sorted alphabetically. The full config is still saved to YAML separately.
    """
    keys = keys or sorted(config)
    width = 60
    print("=" * width)
    print(f"  {title}")
    print("-" * width)
    for k in keys:
        if k in config:
            print(f"  {k:<22}: {config[k]}")
    print("=" * width)


def write_generation_config(out_dir: str, config: dict, filename="generation_config.yaml") -> str:
    """Write ``config`` to ``out_dir/filename`` as YAML and return the path."""
    import yaml

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=True)
    return path
