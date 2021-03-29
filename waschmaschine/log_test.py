import logging_cfg
import logging
import log_test2

logging_cfg.init_config()
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    logger.debug("neu")
    log_test2.print_hello()
