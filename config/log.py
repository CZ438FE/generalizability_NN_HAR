import logging
import logging.config


def get_log_levels():
    """returns a dict which maps strings of log-levels to log-level-constants
    """
    return {
        'critical': logging.CRITICAL,
        'error': logging.ERROR,
        'warn': logging.WARNING,
        'warning': logging.WARNING,
        'info': logging.INFO,
        'debug': logging.DEBUG
    }


def setup_logger(name, loglevel=None):
    """initializes logger with given loglevel
    """
    global logger
    logging.config.fileConfig(fname='config/logging.conf')
    logger = logging.getLogger(name)
    if loglevel:
        logger.setLevel(get_log_levels()[loglevel])
    return logger


logger = logging.getLogger('main')
