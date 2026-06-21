"""Configuration loading utilities."""
import re
import yaml
import os


def _expand_vars_with_defaults(s: str) -> str:
    """Expand ${VAR:-default} and ${VAR} patterns in a string."""
    def _replace(m):
        var = m.group(1)
        default = m.group(2)
        val = os.environ.get(var)
        if val is not None:
            return val
        return default if default is not None else m.group(0)
    return re.sub(r'\$\{([^}:]+)(?::-([^}]*))?\}', _replace, s)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file with environment variable expansion."""
    with open(config_path, "r", encoding="utf-8") as f:
        raw = f.read()
    expanded = _expand_vars_with_defaults(raw)
    cfg = yaml.safe_load(expanded)
    return cfg


def merge_configs(base: dict, override: dict) -> dict:
    """Merge override into base config (shallow)."""
    merged = base.copy()
    merged.update(override)
    return merged
