"""Allow ``python -m tribe_pipeline ...`` to invoke the CLI."""

from tribe_pipeline.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
