import argparse
import logging
import sys

import numpy as np
import tqdm

from bbert.model.statistics import BBLModel
from bbert.utils.bb_splitter import split_into_bbs


class Inference:
    def __init__(self):
        self.bbl = BBLModel('bb_stat.npy')

    def analyze(self, bbs):
        bbl_score = self.bbl.calc_score(bbs)
        logging.info("BBL score: {}".format(bbl_score))


def run_one_file(target_filename, inference: Inference):
    with open(target_filename, 'rt') as f:
        instructions = [line.strip() for line in f.readlines()]
        logging.info('load file {} (total sequence: {})'.format(args.input, len(instructions)))

    if len(instructions) <= 100:
        logging.error('instruction sequence is too short. abort')
        return None

    bbs = split_into_bbs(instructions)
    logging.info('the number of basic blocks: {}'.format(len(bbs)))

    inference.analyze(bbs)


def main(args):
    inference = Inference()

    run_one_file(args.input, inference)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='the filename of file to analze')

    logging.basicConfig(format='%(asctime)s %(message)s')
    logging.getLogger().setLevel(logging.INFO)

    args = parser.parse_args()
    main(args)
