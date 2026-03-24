"""
Config loader for Allama configuration files.
Each physical and logical model has its own .all file.
"""
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

logger = logging.getLogger("allama.config_loader")


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

            if current_section is not None:
                current_section[key] = value
            else:
                result[key] = value

    return result


def load_models_from_configs(
    config_dir: str = "configs"
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Load all .all configuration files from a directory.

    Naming conventions:
    - Physical models: <name>.all (e.g., qwen3.5-27b.all)
      Contains: backend, path/model, and all physical model settings

    - Logical models: <name>.all (e.g., qwen3.5-27b-instruct.all)
      Contains: physical (reference to physical model name), plus sampling overrides

    Args:
        config_dir: Path to directory containing .all files

    Returns:
        Tuple of (physical_models, logical_models)
    """
    config_path = Path(config_dir)

    if not config_path.exists():
        logger.warning(f"Config directory not found: {config_dir}")
        return {}, {}

    physical_models: Dict[str, Dict[str, Any]] = {}
    logical_models: Dict[str, Dict[str, Any]] = {}

    # Load from physical subdirectory
    physical_dir = config_path / "physical"
    if physical_dir.exists():
        for cfg_file in physical_dir.glob("*"):
            if cfg_file.is_file() and cfg_file.suffix in (".all", ".allm"):
                try:
                    logger.debug(f"Loading physical config: {cfg_file.name}")
                    content = cfg_file.read_text(encoding='utf-8')
                    config = parse_all_file(content)

                    if "backend" in config:
                        model_name = cfg_file.stem
                        physical_models[model_name] = config
                        logger.debug(f"  Loaded physical model: {model_name}")
                    else:
                        logger.warning(f"  Skipped {cfg_file.name}: no 'backend' field")

                except Exception as e:
                    logger.error(f"Failed to load {cfg_file.name}: {e}")

    # Load from logical subdirectory
    logical_dir = config_path / "logical"
    if logical_dir.exists():
        for cfg_file in logical_dir.glob("*"):
            if cfg_file.is_file() and cfg_file.suffix in (".all", ".allm"):
                try:
                    logger.debug(f"Loading logical config: {cfg_file.name}")
                    content = cfg_file.read_text(encoding='utf-8')
                    config = parse_all_file(content)

                    if "physical" in config:
                        model_name = cfg_file.stem
                        physical_ref = config["physical"]

                        if physical_ref not in physical_models:
                            logger.warning(
                                f"Logical model '{model_name}' references unknown "
                                f"physical model '{physical_ref}'"
                            )

                        logical_models[model_name] = config
                        logger.debug(f"  Loaded logical model: {model_name} -> {physical_ref}")
                    else:
                        logger.warning(f"  Skipped {cfg_file.name}: no 'physical' field")

                except Exception as e:
                    logger.error(f"Failed to load {cfg_file.name}: {e}")

    return physical_models, logical_models
