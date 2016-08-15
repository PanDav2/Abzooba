# coding: utf8
import datetime
import logging
import os
import sys


def init_logging():
    day = format(datetime.datetime.now().replace(second=0, microsecond=0))
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    a = os.getcwd() + '/logs/' + __file__.split('/')[-1:][0]
    file_handler = logging.FileHandler('{}__{}.log'.format(a, day))
    file_handler.setFormatter(logging.Formatter('%(levelname)s - %(asctime)s - %(message)s'))
    root.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter('%(levelname)s - %(asctime)s - %(message)s'))
    root.addHandler(stream_handler)
