import logging
import sys


def setup_mas_logger(name="MAS_SYSTEM"):
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        logger.propagate = False

    return logger

mas_logger = setup_mas_logger()