"""
Configuration loader for RL training system.
Loads YAML files with backend-specific variants and `base_config` inheritance.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Set, Union

import yaml


class ConfigDict:
    """Dictionary-like config object with dot notation access."""

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            return object.__getattribute__(self, key)
        try:
            value = self._data[key]
            if isinstance(value, dict):
                return ConfigDict(value)
            return value
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        if isinstance(value, dict):
            return ConfigDict(value)
        return value

    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default fallback."""
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to plain dictionary."""
        return self._data

    def update(self, other: Union[Dict, "ConfigDict"]) -> None:
        """Update config with another dict or ConfigDict."""
        if isinstance(other, ConfigDict):
            other = other.to_dict()
        self._data.update(other)

    def __repr__(self) -> str:
        return f"ConfigDict({self._data})"


class ConfigLoader:
    """Loads and manages YAML configuration files."""

    def __init__(self, config_dir: Optional[str] = None):
        if config_dir is None:
            config_dir = Path.cwd() / "rl_framework" / "rl_configs"

        self.config_dir = Path(config_dir)
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {self.config_dir}")

        self._configs: Dict[str, ConfigDict] = {}
        self._backend_aliases = {"uniad", "vad"}

    @staticmethod
    def _deep_update(base: Dict, override: Dict) -> Dict:
        """Recursively update nested dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader._deep_update(result[key], value)
            else:
                result[key] = value
        return result

    def _read_yaml_dict(self, yaml_path: Path) -> Dict[str, Any]:
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config file must contain a mapping at top-level: {yaml_path}")
        return data

    def _load_yaml_with_base(self, yaml_path: Path, visiting: Optional[Set[Path]] = None) -> Dict[str, Any]:
        yaml_path = yaml_path.resolve()
        visiting = visiting or set()
        if yaml_path in visiting:
            raise ValueError(f"Circular base_config reference detected at: {yaml_path}")

        visiting.add(yaml_path)
        data = self._read_yaml_dict(yaml_path)
        base_ref = data.pop("base_config", None)
        if base_ref is not None:
            if not isinstance(base_ref, str) or not base_ref.strip():
                raise ValueError(f"`base_config` must be a non-empty string in {yaml_path}")
            base_path = Path(base_ref)
            if not base_path.is_absolute():
                base_path = (yaml_path.parent / base_path).resolve()
            base_data = self._load_yaml_with_base(base_path, visiting)
            data = self._deep_update(base_data, data)
        visiting.remove(yaml_path)
        return data

    def load(self, config_name: str, override_path: Optional[str] = None) -> ConfigDict:
        """
        Load a YAML config file with optional `base_config` inheritance.

        Args:
            config_name: Name of config file (without .yaml extension)
            override_path: Optional path to override config file
        """
        config_path = self.config_dir / f"{config_name}.yaml"
        config_data = self._load_yaml_with_base(config_path)

        if override_path is not None:
            override_file = Path(override_path)
            if override_file.exists():
                override_data = self._load_yaml_with_base(override_file)
                config_data = self._deep_update(config_data, override_data)

        config = ConfigDict(config_data)
        self._configs[config_name] = config
        return config

    def _validate_backend(self, backend: str) -> str:
        backend = str(backend).strip().lower()
        if backend not in self._backend_aliases:
            raise ValueError(
                f"Unsupported backend '{backend}', expected one of: {sorted(self._backend_aliases)}"
            )
        return backend

    def load_all(self, override_dir: Optional[str] = None, backend: str = "uniad") -> Dict[str, ConfigDict]:
        """
        Load all standard config files for a backend.

        Args:
            override_dir: Optional directory containing override YAML files
            backend: Backend name, one of {"uniad", "vad"}
        """
        backend = self._validate_backend(backend)
        config_plan = {
            "ppo_training": f"ppo_training/{backend}",
            "policy_network": f"policy_network/{backend}",
            "environment": f"environment/{backend}",
            "paths": f"paths/{backend}",
        }

        for logical_name, physical_name in config_plan.items():
            override_path = None
            if override_dir is not None:
                candidate = Path(override_dir) / f"{physical_name}.yaml"
                if candidate.exists():
                    override_path = str(candidate)

            config = self.load(physical_name, override_path)
            self._configs[logical_name] = config
            self._configs[physical_name] = config

        return self._configs

    def get(self, config_name: str) -> ConfigDict:
        """Get a loaded config by name."""
        if config_name not in self._configs:
            raise KeyError(f"Config '{config_name}' not loaded. Call load() first.")
        return self._configs[config_name]


def load_configs(
    config_dir: Optional[str] = None,
    override_dir: Optional[str] = None,
    backend: str = "uniad",
) -> Dict[str, ConfigDict]:
    """Convenience function to load all configs."""
    loader = ConfigLoader(config_dir)
    return loader.load_all(override_dir, backend=backend)
