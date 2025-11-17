import logging

# usage: 
# logger = logging.getLogger(__name__)
# call this at the top of any script which needs logging.
# IMPORTANT: call setup_logging() once (and only once) at the start of your main script to configure logging.

def setup_logging(level=logging.INFO):
    """
    Project-level logging setup function.
    Configures the root logger with a standard format and level.
    """

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # noisy libraries can be set to WARNING level here if required
    for noisy in []:
        logging.getLogger(noisy).setLevel(logging.WARNING)