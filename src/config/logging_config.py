import logging

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # noisy libraries can be set to WARNING level here if required
    for noisy in []:
        logging.getLogger(noisy).setLevel(logging.WARNING)