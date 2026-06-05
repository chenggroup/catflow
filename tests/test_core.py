"""Test catflow.core adapter module (ai2-kit re-exports)."""


class TestCoreImports:
    def test_apply_checkpoint_importable(self):
        """@apply_checkpoint can be imported from catflow.core."""
        from catflow.core import apply_checkpoint

        assert callable(apply_checkpoint)

    def test_set_checkpoint_dir_importable(self):
        """set_checkpoint_dir can be imported and called."""
        from catflow.core import set_checkpoint_dir

        assert callable(set_checkpoint_dir)

    def test_artifact_importable(self):
        """Artifact class can be imported."""
        from catflow.core import Artifact

        assert Artifact is not None

    def test_bash_script_importable(self):
        """BashScript and BashStep can be imported."""
        from catflow.core import BashScript, BashStep

        assert BashScript is not None
        assert BashStep is not None
