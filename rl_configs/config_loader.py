"""
Configuration loader for RL training system.
Loads YAML configuration files with support for nested access and backend variants.
"""
import yaml
from typing import Any, Dict, Optional, Union
from pathlib import Path


class ConfigDict:
    """Dictionary-like config object with dot notation access."""

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def __getattr__(self, key: str) -> Any:
        if key.startswith('_'):
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

    def update(self, other: Union[Dict, 'ConfigDict']) -> None:
        """Update config with another dict or ConfigDict."""
        if isinstance(other, ConfigDict):
            other = other.to_dict()
        self._data.update(other)

    def __repr__(self) -> str:
        return f"ConfigDict({self._data})"


class ConfigLoader:
    """Loads and manages YAML configuration files."""

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize config loader.

        Args:
            config_dir: Directory containing YAML config files.
                       Defaults to rl_configs/ in project root.
        """
        if config_dir is None:
            # Default to rl_configs directory
            project_root = Path(__file__).parent.parent
            config_dir = project_root / "rl_configs"

        self.config_dir = Path(config_dir)
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {self.config_dir}")

        self._configs: Dict[str, ConfigDict] = {}
        self._backend_aliases = {"uniad", "vad"}

    def load(self, config_name: str, override_path: Optional[str] = None) -> ConfigDict:
        """
        Load a YAML config file.

        Args:
            config_name: Name of config file (without .yaml extension)
            override_path: Optional path to override config file

        Returns:
            ConfigDict object with loaded configuration
        """
        # Load base config
        config_path = self.config_dir / f"{config_name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f) or {}

        # Apply overrides if provided
        if override_path is not None:
            override_path = Path(override_path)
            if override_path.exists():
                with open(override_path, 'r', encoding='utf-8') as f:
                    override_data = yaml.safe_load(f) or {}
                config_data = self._deep_update(config_data, override_data)

        config = ConfigDict(config_data)
        self._configs[config_name] = config
        return config

    def _resolve_variant_name(self, logical_name: str, backend: str) -> str:
        backend = str(backend).strip().lower()
        if backend not in self._backend_aliases:
            raise ValueError(f"Unsupported backend '{backend}', expected one of: {sorted(self._backend_aliases)}")

        variant_name = f"{logical_name}_{backend}"
        variant_path = self.config_dir / f"{variant_name}.yaml"
        if variant_path.exists():
            return variant_name

        # Backward compatibility: allow legacy non-suffixed file names.
        return logical_name

    def load_all(self, override_dir: Optional[str] = None, backend: str = "uniad") -> Dict[str, ConfigDict]:
        """
        Load all standard config files.

        Args:
            override_dir: Optional directory containing override YAML files
            backend: Backend name, one of {"uniad", "vad"}

        Returns:
            Dictionary mapping config names to ConfigDict objects
        """
        config_plan = {
            "ppo_training": "ppo_training",
            "policy_network": self._resolve_variant_name("policy_network", backend),
            "environment": self._resolve_variant_name("environment", backend),
            "paths": "paths",
        }

        for logical_name, physical_name in config_plan.items():
            override_path = None
            if override_dir is not None:
                # Prefer backend-specific override files, then fallback to logical name.
                candidate_paths = [
                    Path(override_dir) / f"{physical_name}.yaml",
                    Path(override_dir) / f"{logical_name}.yaml",
                ]
                for candidate in candidate_paths:
                    if candidate.exists():
                        override_path = candidate
                        break

            config = self.load(physical_name, override_path)
            self._configs[logical_name] = config
            # Also cache physical name for direct lookups.
            self._configs[physical_name] = config

        return self._configs

    def get(self, config_name: str) -> ConfigDict:
        """Get a loaded config by name."""
        if config_name not in self._configs:
            raise KeyError(f"Config '{config_name}' not loaded. Call load() first.")
        return self._configs[config_name]

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


def load_configs(
    config_dir: Optional[str] = None,
    override_dir: Optional[str] = None,
    backend: str = "uniad",
) -> Dict[str, ConfigDict]:
    """
    Convenience function to load all configs.

    Args:
        config_dir: Base config directory (default: rl_configs/)
        override_dir: Optional override directory
        backend: Backend name, one of {"uniad", "vad"}

    Returns:
        Dictionary with keys: 'ppo_training', 'policy_network', 'environment', 'paths'
    """
    loader = ConfigLoader(config_dir)
    return loader.load_all(override_dir, backend=backend)
