"""
Config loader for Allma configuration files.
Each base model and profile has its own .all file.
"""
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("allma.config_loader")


def parse_all_file(content: str) -> Dict[str, Any]:
    """Parse a single config file content."""
    import json
    result = {}
    current_section = None
    lines = content.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        i += 1

        if not line or line.startswith('#'):
            continue

        # Section header [section_name]
        if line.startswith('[') and line.endswith(']') and '=' not in line:
            section_name = line[1:-1].strip()
            result[section_name] = {}
            current_section = result[section_name]
            continue

        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()

            # Multi-line list support
            if value.startswith('['):
                accumulated = value
                while not accumulated.rstrip().endswith(']') and i < len(lines):
                    accumulated += ' ' + lines[i].strip()
                    i += 1
                try:
                    parsed = json.loads(accumulated)
                    if current_section is not None:
                        current_section[key] = parsed
                    else:
                        result[key] = parsed
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse list value for key '{key}': {accumulated}")
                continue

            # Remove quotes
            value = value.strip('"').strip("'")

            # Type coercion
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value.lower() in ('null', 'none', '~'):
                value = None
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass

            if current_section is not None:
                current_section[key] = value
            else:
                result[key] = value

    return result


def load_models_from_configs(
    config_dir: str = "configs"
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Load all .all configuration files from a directory.

    Naming conventions:
    - Base models: <name>.all (e.g., qwen3.5-27b.all)
      Contains: backend, path/model, and all base model settings

    - Profiles: <name>.all (e.g., qwen3.5-27b-instruct.all)
      Contains: base (reference to base model name), plus sampling overrides

    Args:
        config_dir: Path to directory containing .all files

    Returns:
        Tuple of (base_models, profile_models)
    """
    config_path = Path(config_dir)

    if not config_path.exists():
        logger.warning(f"Config directory not found: {config_dir}")
        return {}, {}

    base_models: Dict[str, Dict[str, Any]] = {}
    profile_models: Dict[str, Dict[str, Any]] = {}

    # Load from base subdirectory
    base_dir = config_path / "base"
    if base_dir.exists():
        for cfg_file in base_dir.glob("*"):
            if cfg_file.is_file() and cfg_file.suffix in (".all", ".allm"):
                try:
                    logger.debug(f"Loading base config: {cfg_file.name}")
                    content = cfg_file.read_text(encoding='utf-8')
                    config = parse_all_file(content)

                    if "backend" in config:
                        model_name = config.get("name") or cfg_file.stem
                        base_models[model_name] = config
                        logger.debug(f"  Loaded base model: {model_name}")
                    else:
                        logger.warning(f"  Skipped {cfg_file.name}: no 'backend' field")

                except Exception as e:
                    logger.error(f"Failed to load {cfg_file.name}: {e}")

    # Load from profile subdirectory
    profile_dir = config_path / "profile"
    if profile_dir.exists():
        for cfg_file in profile_dir.glob("*"):
            if cfg_file.is_file() and cfg_file.suffix in (".all", ".allm"):
                try:
                    logger.debug(f"Loading profile config: {cfg_file.name}")
                    content = cfg_file.read_text(encoding='utf-8')
                    config = parse_all_file(content)

                    if "base" in config:
                        model_name = config.get("name") or cfg_file.stem
                        base_ref = config["base"]

                        if base_ref not in base_models:
                            logger.warning(
                                f"Profile '{model_name}' references unknown "
                                f"base model '{base_ref}'"
                            )

                        profile_models[model_name] = config
                        logger.debug(f"  Loaded profile: {model_name} -> {base_ref}")
                    else:
                        logger.warning(f"  Skipped {cfg_file.name}: no 'base' field")

                except Exception as e:
                    logger.error(f"Failed to load {cfg_file.name}: {e}")

    return base_models, profile_models
