"""Bump the project version, record the release in CHANGELOG.md, tag it.

Called by `just release VERSION`. Keeps the Python package
(`pyproject.toml`), the frontend package (`frontend/package.json`) and the
changelog section header all pointing at the same version so the release
workflow finds a `## [vX.Y.Z]` block to extract release notes from.

Deliberately a tiny stdlib-only script -- it runs during `just release` on
whatever Python is handy, before the CI has a chance to pin anything.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-+][\w.-]+)?$")
ROOT = Path(__file__).resolve().parent.parent


def _run(*cmd: str) -> str:
    """Run a command, raise with its stderr on failure, return stdout."""
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"$ {' '.join(cmd)}\n{result.stderr.strip()}")
    return result.stdout.strip()


def _bump_pyproject(version: str) -> None:
    """Replace the first `version = "..."` line in pyproject.toml.

    We don't rewrite the TOML (tomllib is read-only, and tomli-w isn't a
    stdlib dependency): the top-of-file `version = "..."` line is unique
    and stable enough to anchor on."""
    path = ROOT / "pyproject.toml"
    text = path.read_text()
    new_text, count = re.subn(
        r'^version\s*=\s*"[^"]+"', f'version = "{version}"', text, count=1, flags=re.M
    )
    if count != 1:
        raise SystemExit(f"could not find the version line in {path}")
    path.write_text(new_text)


def _bump_package_json(version: str) -> None:
    path = ROOT / "frontend" / "package.json"
    data = json.loads(path.read_text())
    data["version"] = version
    path.write_text(json.dumps(data, indent=2) + "\n")


def _prepend_changelog(version: str) -> None:
    """Insert a `## [vX.Y.Z] - YYYY-MM-DD` section at the top of CHANGELOG.md.

    The release workflow greps for this header to pull release notes. The
    existing changelog is a flat bullet list; we promote the current top
    bullets into the new section without otherwise reformatting."""
    path = ROOT / "CHANGELOG.md"
    text = path.read_text()
    marker = "# Changelog\n\n"
    if not text.startswith(marker):
        raise SystemExit(f"{path} does not start with '# Changelog'")
    body = text[len(marker) :]
    header = f"## [v{version}] - {date.today().isoformat()}\n\n"
    path.write_text(f"{marker}{header}{body}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: cut_release.py X.Y.Z")
    version = sys.argv[1].lstrip("v")
    if not SEMVER_RE.match(version):
        raise SystemExit(f"not a semver version: {version!r}")

    if _run("git", "status", "--porcelain"):
        raise SystemExit("working tree is dirty; commit or stash first")
    branch = _run("git", "rev-parse", "--abbrev-ref", "HEAD")
    if branch != "main":
        raise SystemExit(f"release must be cut from main, not {branch!r}")

    tag = f"v{version}"
    existing = _run("git", "tag", "-l", tag)
    if existing:
        raise SystemExit(f"tag {tag} already exists")

    _bump_pyproject(version)
    _bump_package_json(version)
    _prepend_changelog(version)

    _run("git", "add", "pyproject.toml", "frontend/package.json", "CHANGELOG.md")
    _run("git", "commit", "-m", f"chore(release): {tag}")
    _run("git", "tag", "-a", tag, "-m", tag)
    print(f"Committed and tagged {tag}. Run `git push origin main {tag}` to publish.")


if __name__ == "__main__":
    main()
