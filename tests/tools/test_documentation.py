"""Repository links must work without ignored local build/model artifacts."""

from pathlib import Path
import subprocess
import tempfile
import unittest
from urllib.parse import unquote

from tools.maintenance.check_docs import check


class DocumentationChecks(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory).resolve()
        subprocess.run(["git", "init", "--quiet", str(root)], check=True, capture_output=True)
        (root / ".gitignore").write_text("/build/\n/models/\n/.venv*/\n__pycache__/\n*.pyc\n")
        (root / "docs").mkdir()
        (root / "README.md").write_text("# Project\n")
        for name in ("BENCHMARK", "MODEL_ZOO", "docs/quantization", "docs/visual-examples"):
            for suffix in (".md", "_zh.md"):
                (root / (name + suffix)).write_text("# Documentation\n")
        return root

    def test_ignored_links_fail_whether_or_not_the_local_artifact_exists(self):
        cases = (
            ("build/receipt.json", "[receipt](build/receipt.json)"),
            ("models/reference/manifest.json", "[manifest](models/reference/manifest.json#summary)"),
            (".venv-reference/report.md", "[report](.venv-reference/report.md)"),
            ("docs/__pycache__/receipt.json", "[cache](docs/__pycache__/receipt.json)"),
            ("build/receipt.json", "![image](docs/../build/receipt.json)"),
            ("build/receipt one.json", '[receipt](<build/receipt one.json> "local receipt")'),
            ("build/receipt.json", "[receipt][result]\n\n[result]: b%75ild/receipt.json"),
        )
        for destination, content in cases:
            for exists in (False, True):
                with self.subTest(link=content, exists=exists), tempfile.TemporaryDirectory() as temporary:
                    root = self.fixture(temporary)
                    (root / "README.md").write_text(content + "\n")
                    artifact = root / unquote(destination)
                    if exists:
                        artifact.parent.mkdir(parents=True, exist_ok=True)
                        artifact.write_text("local artifact\n")
                    with self.assertRaisesRegex(ValueError, "ignored local doc target"):
                        check(root)

    def test_source_links_migration_mapping_and_plain_archive_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture(temporary)
            source = root / "apps/image/image.cpp"
            source.parent.mkdir(parents=True)
            source.write_text("// source\n")
            (root / "docs/guide.md").write_text("# Guide\n")
            (root / "README.md").write_text(
                '[guide](<docs/guide.md> "guide")\n[source](examples/image.cpp)\n'
                "[website](https://example.com)\n[mail](mailto:example@example.com)\n"
                "[section](#section)\nLocal archive: `build/receipt.json`.\n")
            check(root)
            source.unlink()
            with self.assertRaisesRegex(ValueError, "missing local doc target"):
                check(root)

    def test_ignored_symlink_to_a_public_file_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture(temporary)
            public = root / "docs/guide.md"
            public.write_text("# Guide\n")
            (root / "build").mkdir()
            (root / "build/guide.md").symlink_to(public)
            (root / "README.md").write_text("[guide](build/guide.md)\n")
            with self.assertRaisesRegex(ValueError, "ignored local doc target"):
                check(root)

    def test_other_repository_documents_are_checked_and_ignored_docs_are_skipped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture(temporary)
            (root / "build").mkdir()
            (root / "build/generated.md").write_text("[private](missing-private-file.json)\n")
            check(root)
            (root / "tools").mkdir()
            (root / "tools/README.md").write_text("[missing](missing-tool.md)\n")
            with self.assertRaisesRegex(ValueError, "missing local doc target.*tools/README.md"):
                check(root)
            (root / "tools/missing-tool.md").write_text("# Tool\n")
            check(root)

    def test_platform_benchmark_tables_keep_bilingual_numeric_checks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture(temporary)
            (root / "benchmarks").mkdir()
            english = root / "benchmarks/Linux-4090.md"
            chinese = root / "benchmarks/Linux-4090_zh.md"
            english.write_text("# Linux\n\n| F32 | 0.505 | 4.777 |\n")
            with self.assertRaisesRegex(ValueError, "missing Chinese measurement document"):
                check(root)
            chinese.write_text("# Linux 性能\n\n| F32 | 0.505 | 4.777 |\n")
            check(root)
            chinese.write_text("# Linux 性能\n\n| F32 | 0.506 | 4.777 |\n")
            with self.assertRaisesRegex(ValueError, "bilingual measurement/artifact table differs: benchmarks/Linux-4090"):
                check(root)


if __name__ == "__main__":
    unittest.main()
