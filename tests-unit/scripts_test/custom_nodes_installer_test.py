import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "install_custom_nodes.py"
SPEC = importlib.util.spec_from_file_location("install_custom_nodes", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
INSTALLER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSTALLER)


class CustomNodeInstallerTest(unittest.TestCase):
    def test_read_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = Path(temporary_directory) / "custom-nodes.yaml"
            manifest.write_text(
                "nodes:\n"
                "  - repo: https://github.com/example/custom-node.git\n"
                f"    commit: {'a' * 40}\n"
                "    pip:\n"
                "      - example-package>=1.0\n",
                encoding="utf-8",
            )

            self.assertEqual(
                INSTALLER.read_manifest(manifest),
                [
                    (
                        "https://github.com/example/custom-node.git",
                        "a" * 40,
                        ["example-package>=1.0"],
                    )
                ],
            )

    def test_read_manifest_rejects_non_exact_commits(self):
        for commit in ("main", "abc123", "g" * 40):
            with self.subTest(commit=commit), tempfile.TemporaryDirectory() as directory:
                manifest = Path(directory) / "custom-nodes.yaml"
                manifest.write_text(
                    "nodes:\n"
                    "  - repo: https://github.com/example/custom-node.git\n"
                    f"    commit: {commit}\n",
                    encoding="utf-8",
                )

                with self.assertRaisesRegex(ValueError, "full 40-character Git commit"):
                    INSTALLER.read_manifest(manifest)

    def test_repository_name_removes_git_suffix(self):
        self.assertEqual(
            INSTALLER.repository_name("https://github.com/example/custom-node.git"),
            "custom-node",
        )

    def test_install_node_checks_out_commit_and_installs_requirements(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = "https://github.com/example/custom-node.git"
            commit = "a" * 40
            destination = Path(temporary_directory) / "custom_nodes"
            destination.mkdir()
            node_directory = destination / "custom-node"

            def fake_run(*command, cwd=None):
                if command[:2] == ("git", "clone"):
                    node_directory.mkdir()
                    (node_directory / "requirements.txt").touch()

            with patch.object(INSTALLER, "run", side_effect=fake_run) as run:
                INSTALLER.install_node(
                    repository, commit, ["example-package>=1.0"], destination
                )

            self.assertEqual(
                run.call_args_list,
                [
                    call(
                        "git",
                        "clone",
                        "--filter=blob:none",
                        "--no-checkout",
                        repository,
                        str(node_directory),
                    ),
                    call("git", "checkout", "--detach", commit, cwd=node_directory),
                    call(
                        INSTALLER.sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "-r",
                        str(node_directory / "requirements.txt"),
                        cwd=node_directory,
                    ),
                    call(
                        INSTALLER.sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "example-package>=1.0",
                        cwd=node_directory,
                    ),
                ],
            )


if __name__ == "__main__":
    unittest.main()
