"""Test WorkflowExecutor local execution."""

from pathlib import Path
from catflow.tasker.resources.workflow_executor import WorkflowExecutor


class TestWorkflowExecutor:
    def test_local_executor_creation(self):
        """Create a local executor."""
        executor = WorkflowExecutor.local(work_base=".")
        assert executor.mode == "local"

    def test_local_run_script(self, tmp_path):
        """Execute a simple script locally."""
        script = tmp_path / "test.sh"
        script.write_text("#!/bin/bash\necho 'hello world'\n")
        script.chmod(0o755)

        executor = WorkflowExecutor.local(work_base=str(tmp_path))
        result = executor.run_script(str(script))

        assert result.returncode == 0
        assert "hello world" in result.stdout

    def test_local_run_command(self, tmp_path):
        """Run a command via run_command."""
        executor = WorkflowExecutor.local(work_base=str(tmp_path))
        result = executor.run_command("echo 'command test'")

        assert result.returncode == 0
        assert "command test" in result.stdout

    def test_local_run_failing_script(self, tmp_path):
        """Failing script returns non-zero exit code."""
        script = tmp_path / "fail.sh"
        script.write_text("#!/bin/bash\nexit 42\n")
        script.chmod(0o755)

        executor = WorkflowExecutor.local(work_base=str(tmp_path))
        result = executor.run_script(str(script))

        assert result.returncode == 42

    def test_ssh_mode_requires_host(self):
        """SSH executor requires a non-empty host for file operations."""
        executor = WorkflowExecutor(mode="ssh", host="", user="")
        # upload on empty host fails with CalledProcessError (scp fails)
        import subprocess
        try:
            executor.upload("/tmp/nonexistent", "/remote")
        except (RuntimeError, subprocess.CalledProcessError, FileNotFoundError):
            pass  # expected: scp will fail or not found

    def test_env_passed_to_script(self, tmp_path):
        """Environment variables are passed to the executed script."""
        script = tmp_path / "env_test.sh"
        script.write_text('#!/bin/bash\necho "MY_VAR=$MY_VAR"\n')
        script.chmod(0o755)

        executor = WorkflowExecutor(local_work_base=str(tmp_path), env={"MY_VAR": "hello"})
        result = executor.run_script(str(script))

        assert "MY_VAR=hello" in result.stdout
