from __future__ import annotations

import sys

from _bootstrap import REPOSITORY_ROOT  # noqa: F401

from scenearchitect.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["doctor", *sys.argv[1:]]))
