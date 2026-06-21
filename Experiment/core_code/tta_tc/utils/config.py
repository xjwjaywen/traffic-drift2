"""Configuration loading utilities."""
import yaml
import os


def load_config(config_path: str) -> dict:
    """Load YAML configuration file with environment variable expansion."""
    with open(config_path, "r", encoding="utf-8") as f:
        raw = f.read()
    expanded = os.path.expandvars(raw)
    cfg = yaml.safe_load(expanded)
    return cfg


def merge_configs(base: dict, override: dict) -> dict:
    """Merge override into base config (shallow)."""
    merged = base.copy()
    merged.update(override)
    return merged
