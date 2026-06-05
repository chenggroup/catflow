"""Test template engine for @VAR@ substitution."""

import os
import tempfile
from pathlib import Path

from catflow.tasker.resources.template_engine import (
    render_template,
    find_template,
    list_template_dir,
)


class TestRenderTemplate:
    def test_simple_substitution(self, tmp_path):
        """Basic @VAR@ replacement works."""
        tpl = tmp_path / "test.tpl"
        tpl.write_text("Hello @NAME@, your @ITEM@ is ready.")

        out = tmp_path / "output.txt"
        render_template(str(tpl), {"NAME": "Alice", "ITEM": "coffee"}, str(out))

        assert out.read_text().strip() == "Hello Alice, your coffee is ready."

    def test_multiple_occurrences(self, tmp_path):
        """Same variable appearing multiple times is replaced everywhere."""
        tpl = tmp_path / "test.tpl"
        tpl.write_text("@X@ + @X@ = @Y@")

        out = tmp_path / "output.txt"
        render_template(str(tpl), {"X": "2", "Y": "4"}, str(out))

        assert out.read_text().strip() == "2 + 2 = 4"

    def test_unmatched_variable_preserved(self, tmp_path):
        """Unmatched @VAR@ is left as-is (not substituted)."""
        tpl = tmp_path / "test.tpl"
        tpl.write_text("Keep @THIS@, replace @THAT@")

        out = tmp_path / "output.txt"
        render_template(str(tpl), {"THAT": "done"}, str(out))

        content = out.read_text().strip()
        assert "@THIS@" in content
        assert "done" in content

    def test_empty_variables(self, tmp_path):
        """Empty variable dict leaves template unchanged."""
        tpl = tmp_path / "test.tpl"
        tpl.write_text("no vars here")

        out = tmp_path / "output.txt"
        render_template(str(tpl), {}, str(out))

        assert out.read_text().strip() == "no vars here"

    def test_output_directory_created(self, tmp_path):
        """Output directory is created if it doesn't exist."""
        tpl = tmp_path / "test.tpl"
        tpl.write_text("test")

        out = tmp_path / "newdir" / "nested" / "output.txt"
        render_template(str(tpl), {"A": "1"}, str(out))

        assert out.exists()
        assert out.read_text().strip() == "test"


class TestFindTemplate:
    def test_find_exact(self, tmp_path):
        """Find exact template path."""
        tpl_dir = tmp_path / "templates" / "deepmd"
        tpl_dir.mkdir(parents=True)
        tpl_file = tpl_dir / "input.json.template"
        tpl_file.write_text("{}")

        os.chdir(tmp_path)
        result = find_template("deepmd/input.json")
        assert result is not None
        assert "deepmd/input.json.template" in result

    def test_find_not_found(self, tmp_path):
        """Non-existent template returns None."""
        os.chdir(tmp_path)
        result = find_template("nonexistent/file")
        assert result is None


class TestListTemplateDir:
    def test_list_templates(self, tmp_path):
        """list_template_dir finds all .template files."""
        (tmp_path / "templates" / "deepmd").mkdir(parents=True)
        (tmp_path / "templates" / "lammps").mkdir(parents=True)
        (tmp_path / "templates" / "deepmd" / "input.json.template").write_text("a")
        (tmp_path / "templates" / "lammps" / "run.sh.template").write_text("b")

        os.chdir(tmp_path)
        templates = list_template_dir("templates")
        assert len(templates) >= 2
        assert any("deepmd/input.json" in k for k in templates)
        assert any("lammps/run.sh" in k for k in templates)
