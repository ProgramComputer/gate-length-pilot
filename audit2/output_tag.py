from __future__ import annotations

import os
import re


def tagged(name: str) -> str:
    tag = os.environ.get("AUDIT2_CHECK_TAG", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]*", tag):
        raise ValueError("AUDIT2_CHECK_TAG may contain only letters, digits, underscore and hyphen")
    stem, dot, ext = name.rpartition(".")
    return stem + tag + dot + ext if dot else name + tag
