"""Keep Efficiency print logs off the MCP stdout transport."""

import builtins
from pathlib import Path
import runpy
import sys


def main():
    target = Path(sys.argv[1]).resolve(strict=True)
    original_print = builtins.print

    def log_print(*args, **kwargs):
        if kwargs.get("file") is None or kwargs.get("file") is sys.stdout:
            kwargs["file"] = sys.stderr
        return original_print(*args, **kwargs)

    builtins.print = log_print
    sys.argv = [str(target), *sys.argv[2:]]
    sys.path.insert(0, str(target.parent))
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
