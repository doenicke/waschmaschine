"""Logging configuration

Usage:

    # Run once at startup:
    logging.config.dictConfig(LOGGING_CONFIG)

    # Include in each module:
    log = logging.getLogger(__name__)
    log.debug("Logging is configured.")

Formatter examples:

    '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    '%(filename)s:%(lineno)s %(funcName)s() - %(levelname)s %(message)s'

"""
# https://stackoverflow.com/questions/7507825/where-is-a-complete-example-of-logging-config-dictconfig

import logging.config


LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'standard': {
            'format': '%(filename)s:%(lineno)s %(funcName)s() - %(levelname)s %(message)s'
        },
    },
    'handlers': {
        'default': {
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',  # Default is stderr
        },
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['default'],
            'level': 'WARNING',
            'propagate': False
        },
        'log_test2': {
            'handlers': ['default'],
            'level': 'DEBUG',
            'propagate': False
        },
        '__main__': {  # if __name__ == '__main__'
            'handlers': ['default'],
            'level': 'DEBUG',
            'propagate': False
        },
    }
}


def init_config():
    logging.config.dictConfig(LOGGING_CONFIG)
