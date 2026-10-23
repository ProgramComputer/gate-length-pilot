from __future__ import annotations

from pathlib import Path
import hashlib
import json
import subprocess
from output_tag import tagged


def main() -> None:
    root = Path(__file__).resolve().parent
    repo = root.parent
    before = json.loads((root / "phase_c_checks/tracked_before.json").read_text())
    names = (
        subprocess.check_output(["git", "--no-optional-locks", "ls-files", "-z"], cwd=repo)
        .decode()
        .split("\0")
    )
    names = [p for p in names if p]
    assert set(names) == set(before)
    after: dict[str, str] = {}
    for name in names:
        with (repo / name).open("rb") as f:
            after[name] = hashlib.file_digest(f, "sha256").hexdigest()
    changed = [p for p in before if before[p] != after[p]]
    status = subprocess.run(
        ["git", "--no-optional-locks", "status", "--short", "--untracked-files=normal"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    script = root / "independent_nope.py"
    with script.open("rb") as f:
        current_hash = hashlib.file_digest(f, "sha256").hexdigest()
    run = json.loads((root / "cpu_run/manifest.json").read_text())
    result = {
        "original_tracked_files": len(before),
        "changed": changed,
        "tracked_after": after,
        "main_independent_script_unchanged_since_owner_run": current_hash == run["script_sha256"],
        "git_status": status.stdout,
        "git_status_stderr": status.stderr,
    }
    assert not changed and result["main_independent_script_unchanged_since_owner_run"]
    with (root / tagged("final_integrity.json")).open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "tracked_after"}, indent=2))


if __name__ == "__main__":
    main()
