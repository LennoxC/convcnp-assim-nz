import logging
import os
import sys
from pathlib import Path
from convcnp_assim_nz.config.env_loader import get_env_var

# usage:
# logger = logging.getLogger(__name__)
# call this at the top of any script which needs logging.
# IMPORTANT: call setup_logging() once (and only once) at the start of your main script to configure logging.


def setup_logging(level=logging.INFO, log_file: str | None = None):
    """
    Project-level logging setup function.
    Configures the root logger with a standard format and level.

    Adds a `FileHandler` writing to `log_file` when provided; otherwise uses
    the environment variable `NOTEBOOK_LOG_FILE` or defaults to
    `<project_root>/.logs.out`.
    """

    # resolve default log file: env var -> explicit -> project-root .logs.out
    if log_file is None:
        log_file = get_env_var("NOTEBOOK_LOG_FILE")

    if log_file is None:
        # logging_config.py lives in src/config; project root is two parents up
        project_root = Path(__file__).resolve().parents[2]
        log_file = str(project_root / ".logs.out")

    handlers = []

    # always keep the stream handler so logs also appear in stderr
    handlers.append(logging.StreamHandler(sys.stderr))

    # add a file handler
    try:
        # ensure parent directory exists (useful when run_notebook_nohup.sh sets a logs/ dir)
        try:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(level)
        handlers.append(fh)
    except Exception:
        # fallback: log a warning to stderr (handlers not yet configured)
        sys.stderr.write(f"Warning: could not create log file handler for {log_file}\n")

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )

    # noisy libraries can be set to WARNING level here if required
    for noisy in []:
        logging.getLogger(noisy).setLevel(logging.WARNING)