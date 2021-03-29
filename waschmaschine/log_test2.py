import logging_cfg
import logging.config

logging_cfg.init_config()
logger = logging.getLogger(__name__)

# logger.setLevel(logging.DEBUG)


def print_hello():
    logger.info("info from print_hello()")
    logger.debug("debug from print_hello()")
    logger.warning("warning from print_hello()")

    print("Hello!")


if __name__ == '__main__':
    print(__name__)
    logger.info("info from module")
    logger.debug("debug from module")
    logger.warning("warning from module")

    print_hello()
