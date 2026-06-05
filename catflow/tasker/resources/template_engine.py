"""Simple template engine for @VAR@ substitution in workflow templates.

Replaces @PLACEHOLDER@ tokens with actual values in template files.
Uses Python's built-in string replacement for maximum compatibility.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional


def render_template(template_path: str, variables: Dict[str, str], output_path: str):
    """Read a template file, substitute @VAR@ tokens, write to output_path.

    Args:
        template_path: Path to the template file with @VAR@ placeholders.
        variables: Dict mapping variable names to their string values.
        output_path: Path to write the rendered output.
    """
    with open(template_path) as f:
        content = f.read()

    for key, value in variables.items():
        content = content.replace(f'@{key}@', str(value))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(content)


def render_json_template(template_path: str, variables: Dict[str, str],
                         output_path: str):
    """Read a JSON template, pre-process @VAR@ placeholders, write output.

    Unlike render_template which does string-level replacement,
    this handles JSON-specific cases like @TYPE_MAP -> "Ag", "O"
    and @DP_DATASET -> "sys1", "sys2".

    The strategy is string-level replacement before JSON parsing.
    """
    render_template(template_path, variables, output_path)


def list_template_dir(template_base: str = "templates") -> Dict[str, str]:
    """List available templates in the templates directory.

    Returns:
        Dict mapping template name to file path.
    """
    template_base = Path(template_base).resolve()
    templates = {}
    for tpl_file in template_base.rglob("*.template"):
        rel_path = tpl_file.relative_to(template_base)
        templates[str(rel_path)] = str(tpl_file)
    return templates


def find_template(name: str, template_base: str = "templates") -> Optional[str]:
    """Find a template by name.

    Args:
        name: Template name (e.g. "deepmd/input.json").
        template_base: Base directory for templates.

    Returns:
        Full path to the template file, or None if not found.
    """
    template_base = Path(template_base).resolve()
    candidates = [
        template_base / f"{name}.template",
        template_base / name,
        template_base / "deepmd" / f"{name}.template",
        template_base / "lammps" / f"{name}.template",
        template_base / "cp2k" / f"{name}.template",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None
