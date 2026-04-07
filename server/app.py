"""
OpenEnv entrypoint shim.

The real FastAPI app lives in `sql_optimizer.server.app`. OpenEnv's validator
(and its default uvicorn target `server.app:app`) expects a top-level
`server/app.py` module at the repo root, so this file simply re-exports it.
"""

from sql_optimizer.server.app import app
from sql_optimizer.server.app import main as _main

__all__ = ["app", "main"]


def main() -> None:
    """Start the OpenEnv FastAPI server (delegates to sql_optimizer.server.app.main)."""
    _main()


if __name__ == "__main__":
    main()
