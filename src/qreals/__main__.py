"""Allow `python -m qreals` to open the same interface as the `qreals` command."""

from .app import main

if __name__ == "__main__":
    raise SystemExit(main())
