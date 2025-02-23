from logging.config import dictConfig

PROD_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'loggers': {
        '': {
            'handlers': ['console'],
            'level': 'INFO',
        },
        'trades': {
            'handlers': ['file_trades'],
            'level': 'WARNING',
        },
    },
    'handlers': {
        'console': {
            # 'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            # 'formatter': 'brief',
            # 'filters': ['my_filter'],
            # 'stream': 'ext://sys.stdout'
        },
        'file_trades': {
            # 'level': 'INFO',
            'class': 'logging.FileHandler',
            # 'formatter': 'verbose',
            'filename': 'trades.log',
            # 'mode': 'a',
            # 'encoding': 'utf-8',
        },
    }
}

TEST_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'loggers': {
        '': {
            'handlers': ['console'],
            'level': 'INFO',
        },
        'trades': {
            'propagate': False,
            'handlers': ['file_trades'],
            'level': 'DEBUG',
        },
    },
    'handlers': {
        'console': {
            # 'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            # 'formatter': 'brief',
            # 'filters': ['my_filter'],
            # 'stream': 'ext://sys.stdout'
        },
        'file_trades': {
            # 'level': 'INFO',
            'class': 'logging.FileHandler',
            # 'formatter': 'verbose',
            'filename': 'trades.log',
            # 'mode': 'a',
            # 'encoding': 'utf-8',
        },
    }
}