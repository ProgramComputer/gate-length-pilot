from __future__ import annotations

import sys

sys.dont_write_bytecode = True
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os

os.environ["MPLCONFIGDIR"] = str(ROOT / "audit" / "mplconfig")
import contextlib
import hashlib
import io
import json
import runpy
import subprocess


def tracked_hashes() -> dict[str, str]:
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    return {
        n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest()
        for n in names
        if n and (ROOT / n).is_file()
    }


def main() -> None:
    assert Path.cwd() == ROOT, "Run from repository root"
    out = ROOT / "audit" / "decisions"
    out.mkdir(parents=True, exist_ok=True)
    before = tracked_hashes()
    original = Path.write_text
    summary: dict[str, dict[str, list[str] | bool]] = {}
    for index, name in enumerate(["report.py", "report2.py", "report3.py", "report4.py"], 1):
        expected = ROOT / ("results" if index == 1 else f"results{index}") / "decision.md"
        destination = out / f"pilot{index}-decision.md"

        def redirected(self: Path, data: str, *args: str | None, **kwargs: str | None) -> int:
            if self.resolve() != expected:
                raise RuntimeError(f"Unexpected report write: {self}")
            return original(destination, data, *args, **kwargs)

        stream = io.StringIO()
        old_argv = sys.argv
        try:
            Path.write_text = redirected
            sys.argv = [name, "decision"]
            with contextlib.redirect_stdout(stream):
                runpy.run_path(str(ROOT / name), run_name="__main__")
        finally:
            Path.write_text = original
            sys.argv = old_argv
        original(out / f"pilot{index}-stdout.txt", stream.getvalue(), encoding="utf-8")
        text = destination.read_text(encoding="utf-8")
        saved = expected.read_text(encoding="utf-8")
        verdicts = [line for line in text.splitlines() if "Verdict" in line]
        summary[name] = {"verdict_lines": verdicts, "matches_saved_decision_text": text == saved}
        print(name, summary[name], flush=True)
    after = tracked_hashes()
    assert before == after, "A tracked file changed"
    original(out / "summary.json", json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    original(
        out / "tracked-file-sha256.json", json.dumps(before, indent=2) + "\n", encoding="utf-8"
    )
    print(f"All {len(before)} tracked file hashes unchanged.", flush=True)


if __name__ == "__main__":
    main()
