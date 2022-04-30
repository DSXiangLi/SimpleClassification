# -*-coding:utf-8 -*-

import os
import logging


def get_logger(name, log_dir=None, log_level=logging.INFO):
    logger = logging.Logger(name)
    logger.setLevel(log_level)
    formater = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formater)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    if log_dir:
        # if log dir not provided, use stream handler only.
        handler = logging.FileHandler(os.path.join(log_dir, 'train.log'), 'a')
        handler.setFormatter(formater)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)

    return logger


logger = get_logger('default')