"""What must NEVER be copied into the Modal container, and what the image skips.

Kept free of any ``modal`` import so an ordinary test can check it (the project's
test environment does not install the Modal client). ``image.py`` passes
``IGNORE_PATTERNS`` to ``add_local_dir`` and ``requirements_for_container`` to
``pip_install``.
"""

from __future__ import annotations

from pathlib import Path

# Patterns relative to the repo root, in Modal's dockerignore-style syntax.
IGNORE_PATTERNS = [
    # secrets and credentials -- the API keys come from the Modal Secret, never from a file
    ".env",
    ".env.*",
    "!.env.example",
    ".modal.toml",
    ".modal",
    # version control and tooling
    ".git",
    ".claude",
    ".pytest_cache",
    ".pre-commit-config.yaml",
    "**/__pycache__",
    "**/*.pyc",
    "*.egg-info",
    # bulky or local-only data: stores, results, downloaded corpora, demo files
    "eval_store",
    "eval_results",
    "data",
    "demo_corpus",
    "eval",
    # not needed to run the jobs
    "tests",
    "docs",
]

# The Streamlit UI, test-fixture plotting and git hooks are not used by ingestion or
# the benchmark runner; everything else in requirements.txt is installed AS PINNED.
EXCLUDED_REQUIREMENTS = frozenset({"streamlit", "matplotlib", "pre-commit"})


def requirements_for_container(requirements_path: Path) -> list[str]:
    """Pinned requirement lines, minus comments and the packages the jobs do not use."""
    packages: list[str] = []
    for raw in requirements_path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name = line.split("==", 1)[0].split("[", 1)[0].strip().lower()
        if name not in EXCLUDED_REQUIREMENTS:
            packages.append(line)
    return packages
