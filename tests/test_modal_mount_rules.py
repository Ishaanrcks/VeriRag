"""What the Modal container may and may not receive (no Modal client needed).

The container gets the repository by ``add_local_dir``. The API keys must arrive ONLY
through the Modal Secret, so a ``.env`` (or any Modal token file) in the mount would
be a leak; these tests pin the rules that prevent it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval_modal.mount_rules import EXCLUDED_REQUIREMENTS, IGNORE_PATTERNS, requirements_for_container

REPO = Path(__file__).resolve().parent.parent


def test_credentials_and_local_state_are_never_mounted():
    for must_ignore in (".env", ".env.*", ".modal.toml", ".git", "eval_store", "data", "eval_results"):
        assert must_ignore in IGNORE_PATTERNS, must_ignore
    # the example file (placeholders only) stays, so the container's config is documented
    assert "!.env.example" in IGNORE_PATTERNS


def test_the_code_and_question_data_the_jobs_need_are_not_ignored():
    # patterns are root-relative: ignoring "eval" must not hide src/verirag/eval, and
    # eval_data (the question set) must stay
    assert "eval" in IGNORE_PATTERNS and "src" not in IGNORE_PATTERNS
    assert "eval_data" not in IGNORE_PATTERNS and "scripts" not in IGNORE_PATTERNS
    assert (REPO / "src" / "verirag" / "eval" / "mmlongbench_runner.py").exists()


def test_modal_matcher_agrees_if_the_client_is_installed():
    matcher_module = pytest.importorskip("modal.file_pattern_matcher")
    ignored = matcher_module.FilePatternMatcher(*IGNORE_PATTERNS)
    for path in (".env", ".git/config", "eval_store/x/ingest_log.json", "data/processed/a.jsonl",
                 "tests/test_x.py", "pkg/__pycache__/a.pyc"):
        assert ignored(Path(path)), f"{path} would be mounted"
    for path in ("src/verirag/eval/mmlongbench_runner.py", "scripts/ingest_mmlongbench_tier.py",
                 "eval_data/mmlongbench_subset200_v1canonical.json", ".env.example",
                 "eval_modal/image.py", "requirements.txt"):
        assert not ignored(Path(path)), f"{path} would be missing in the container"


def test_requirements_are_installed_as_pinned_minus_the_unused_ones():
    packages = requirements_for_container(REPO / "requirements.txt")
    names = {p.split("==")[0].lower() for p in packages}
    assert not (names & EXCLUDED_REQUIREMENTS)
    assert {"torch", "transformers", "sentence-transformers", "chromadb", "pymupdf", "openai",
            "langgraph", "tiktoken"} <= names
    assert all("==" in p for p in packages), "every container dependency stays pinned"
    assert not any("#" in p for p in packages), "inline comments are stripped"


def test_comments_and_blank_lines_are_skipped(tmp_path):
    f = tmp_path / "r.txt"
    f.write_text("# header\n\nfoo==1.0  # why\nstreamlit==9\nbar[extra]==2\n")
    assert requirements_for_container(f) == ["foo==1.0", "bar[extra]==2"]


def test_gitignore_covers_modal_local_state_and_the_eval_stores():
    ignored = (REPO / ".gitignore").read_text().splitlines()
    for entry in (".env", ".modal.toml", ".modal/", "eval_store/"):
        assert entry in ignored, entry
