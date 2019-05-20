import logging
import os
import sys
import numpy as np


class txt_logger(object):
    def __init__(self, save_dir, name, filename):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
        if save_dir:
            fh = logging.FileHandler(os.path.join(save_dir, filename))
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        self.logger = logger
        self.info = {}

    def add_scalar(self, tag, value, step=None):

        if tag in self.info:
            self.info[tag].append(value)
        else:
            self.info[tag] = [value]

    def print_info(self, epoch):
        info_line = 'epoch {}: '.format(epoch)
        for i in self.info.keys():
            info = np.array(self.info[i]).mean()
            info_line += i + ':' + str(round(info,4)) + ', '

        print(info_line)
        self.logger.info(
            info_line
        )
        self.info = {}




