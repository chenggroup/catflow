"""Workflow executor for CatFlow.

Provides local and remote (SSH) execution of bash scripts.
Replaces the need for ai2-kit's Executor which has Python 3.13 compatibility issues.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, field

from catflow.utils import logger


@dataclass
class WorkflowExecutor:
    """Execute workflow scripts locally or on remote HPC via SSH.

    Two modes:
    - Local: runs scripts directly via subprocess (use on HPC login nodes)
    - SSH: uploads scripts to remote host and executes via SSH connection

    Usage:
        executor = WorkflowExecutor.local()
        executor.run_script("/path/to/script.sh")

        executor = WorkflowExecutor.ssh(host="login.hpc.org", user="user")
        executor.run_script("remote_script.sh")
    """

    mode: str = "local"  # "local" or "ssh"
    host: str = ""
    user: str = ""
    port: int = 22
    key_path: Optional[str] = None
    remote_work_base: str = ""
    local_work_base: str = ""
    env: dict = field(default_factory=dict)

    @classmethod
    def local(cls, work_base: str = ".") -> "WorkflowExecutor":
        """Create a local executor."""
        return cls(mode="local", local_work_base=work_base)

    @classmethod
    def ssh(cls, host: str, user: str, port: int = 22,
            key_path: Optional[str] = None,
            remote_work_base: str = "",
            local_work_base: str = ".") -> "WorkflowExecutor":
        """Create an SSH executor."""
        return cls(
            mode="ssh",
            host=host,
            user=user,
            port=port,
            key_path=key_path,
            remote_work_base=remote_work_base,
            local_work_base=local_work_base,
        )

    @classmethod
    def from_config(cls, machine_config) -> "WorkflowExecutor":
        """Create executor from a MachineConfig instance."""
        if machine_config.remote_executor:
            # Remote mode
            cfg = machine_config.remote_executor
            return cls.ssh(
                host=cfg.get("host", ""),
                user=cfg.get("user", ""),
                port=cfg.get("port", 22),
                key_path=cfg.get("key_path"),
                remote_work_base=cfg.get("remote_root", ""),
            )
        else:
            # Local mode
            return cls.local()

    def run_script(self, script_path: str,
                   work_dir: Optional[str] = None,
                   timeout: Optional[int] = None,
                   env: Optional[dict] = None) -> subprocess.CompletedProcess:
        """Execute a bash script.

        Args:
            script_path: Path to the script to execute.
            work_dir: Working directory for execution.
            timeout: Optional timeout in seconds.
            env: Additional environment variables.

        Returns:
            subprocess.CompletedProcess with stdout/stderr.
        """
        merged_env = {**self.env, **(env or {})}
        if self.mode == "local":
            return self._run_local(script_path, work_dir, timeout, merged_env)
        else:
            return self._run_remote(script_path, work_dir, timeout, merged_env)

    def _run_local(self, script_path: str, work_dir: Optional[str],
                   timeout: Optional[int],
                   env: dict) -> subprocess.CompletedProcess:
        """Run a script locally."""
        work_dir = work_dir or self.local_work_base
        logger.info(f"Executing locally: {script_path} in {work_dir}")
        result = subprocess.run(
            ["bash", str(script_path)],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, **env},
        )
        if result.returncode != 0:
            logger.error(f"Script failed (rc={result.returncode}):\n{result.stderr}")
        else:
            logger.info(f"Script completed successfully:\n{result.stdout[-500:]}")
        return result

    def _run_remote(self, script_path: str, work_dir: Optional[str],
                    timeout: Optional[int],
                    env: dict) -> subprocess.CompletedProcess:
        """Run a script on remote host via SSH."""
        work_dir = work_dir or self.remote_work_base
        remote_path = f"{work_dir}/{Path(script_path).name}"

        # Build SSH command
        ssh_cmd_parts = ["ssh"]
        if self.port != 22:
            ssh_cmd_parts.extend(["-p", str(self.port)])
        if self.key_path:
            ssh_cmd_parts.extend(["-i", self.key_path])
        ssh_cmd_parts.append(f"{self.user}@{self.host}")

        # Upload script
        upload_cmd = [
            "scp", str(script_path), f"{self.user}@{self.host}:{remote_path}"
        ]
        logger.info(f"Uploading script to {self.host}:{remote_path}")
        subprocess.run(upload_cmd, capture_output=True, text=True, check=True)

        # Build remote execution command
        env_str = " ".join(f"{k}={v}" for k, v in env.items()) if env else ""
        remote_cmd = f"cd {work_dir} && {env_str} bash {remote_path}"
        ssh_cmd = ssh_cmd_parts + [remote_cmd]

        logger.info(f"Executing remotely on {self.host}: {remote_path}")
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        # Cleanup remote script
        subprocess.run(
            ssh_cmd_parts + [f"rm -f {remote_path}"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            logger.error(f"Remote script failed (rc={result.returncode}):\n{result.stderr}")
        else:
            logger.info(f"Remote script completed:\n{result.stdout[-500:]}")
        return result

    def upload(self, local_path: str, remote_path: str):
        """Upload a file or directory to remote host (SSH mode only)."""
        if self.mode != "ssh":
            raise RuntimeError("upload only supported in SSH mode")
        cmd = ["scp", "-r", local_path, f"{self.user}@{self.host}:{remote_path}"]
        subprocess.run(cmd, check=True)

    def download(self, remote_path: str, local_path: str):
        """Download a file or directory from remote host (SSH mode only)."""
        if self.mode != "ssh":
            raise RuntimeError("download only supported in SSH mode")
        cmd = ["scp", "-r", f"{self.user}@{self.host}:{remote_path}", local_path]
        subprocess.run(cmd, check=True)

    def run_command(self, command: str, work_dir: Optional[str] = None,
                    timeout: Optional[int] = None) -> subprocess.CompletedProcess:
        """Run an arbitrary shell command."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write("#!/bin/bash\nset -e\n")
            f.write(command + "\n")
            script_path = f.name
        os.chmod(script_path, 0o755)
        try:
            return self.run_script(script_path, work_dir, timeout)
        finally:
            os.unlink(script_path)
