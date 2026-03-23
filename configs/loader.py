"""
Config loader for Allama .all configuration files.
Each physical and logical model has its own .all file.
"""
from pathlib import Path
from typing import Dict, Any, List, Set
import logging
import os
import re

logger = logging.getLogger("Wrapper")


def parse_all_file(content: str) -> Dict[str, Any]:
    result = {}
    current_section = None

    for line in content.split('\n'):
        line = line.strip()

        if not line or line.startswith('#'):
            continue

        if line.startswith('[') and line.endswith(']'):
            section_name = line[1:-1].strip()
            result[section_name] = {}
            current_section = result[section_name]
            continue

        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()

            # Suporte a listas: ["a", "b", "c"]
            if value.startswith('[') and value.endswith(']'):
                import json
                try:
                    parsed = json.loads(value)
                    if current_section is not None:
                        current_section[key] = parsed
                    else:
                        result[key] = parsed
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse list value for key '{key}': {value}")
                continue

            # Remove aspas
            value = value.strip('"').strip("'")

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

def load_models_from_configs(config_dir: str = "configs") -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
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
        logger.warning(f"Config directory not found: {config_dir} Using empty configs.")
        return {}, {}

    physical_models: Dict[str, Dict[str, Any]] = {}
    logical_models: Dict[str, Dict[str, Any]] = {}

    # Explicitly load from physical and logical subdirectories
    physical_dir = config_path / "physical"
    logical_dir = config_path / "logical"

    if physical_dir.exists():
        for all_file in physical_dir.glob("*.allm"):
            try:
                logger.debug(f"Loading physical config: {all_file.name}")
                content = all_file.read_text(encoding='utf-8')
                config = parse_all_file(content)

                # Determine if physical or logical model
                if "backend" in config:
                    # Physical model - use filename as key (without .all extension)
                    model_name = all_file.stem
                    physical_models[model_name] = config
                    logger.debug(f"  Loaded physical model: {model_name}")
                else:
                    logger.warning(f"  Skipped {all_file.name}: no 'backend' field")

            except Exception as e:
                logger.error(f"Failed to load {all_file.name}: {e}")

    if logical_dir.exists():
        for all_file in logical_dir.glob("*.allm"):
            try:
                logger.debug(f"Loading logical config: {all_file.name}")
                content = all_file.read_text(encoding='utf-8')
                config = parse_all_file(content)

                # Determine if physical or logical model
                if "physical" in config:
                    # Logical model
                    model_name = all_file.stem
                    physical_ref = config["physical"]

                    # Check if reference is valid (optional - just warn)
                    if physical_ref not in physical_models:
                        logger.warning(f"  Logical model '{model_name}' references unknown physical model '{physical_ref}'")

                    logical_models[model_name] = config
                    logger.debug(f"  Loaded logical model: {model_name} -> {physical_ref}")
                else:
                    logger.warning(f"  Skipped {all_file.name}: no 'physical' field")

            except Exception as e:
                logger.error(f"Failed to load {all_file.name}: {e}")

    return physical_models, logical_models
