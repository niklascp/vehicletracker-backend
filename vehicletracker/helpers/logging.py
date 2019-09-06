import os
import logging

def setup_logging(config):
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    logging.config.dictConfig(config['logging'])
