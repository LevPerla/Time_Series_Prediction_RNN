import logging
import logging.config

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,

    'formatters': {
        'default_formatter': {
            'format': '[%(levelname)s] - %(asctime)s - %(message)s'
        },
    },

    'handlers': {
        'stream_handler': {
            'class': 'logging.StreamHandler',
            'level': 'WARNING',
            'formatter': 'default_formatter',
        },
    },

    'loggers': {
        'ts_rnn_logger': {
            'handlers': ['stream_handler'],
            'level': 'DEBUG',
            'propagate': True
        }
    }
}

logging.config.dictConfig(LOGGING_CONFIG)

# Create logger
logger = logging.getLogger("ts_rnn_logger")
