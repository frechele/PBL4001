import argparse
import logging
import sys

import numpy as np
import torch
import tqdm

from bbert.model.statistics import BBLModel
from bbert.model.statistics import BBWModel
from bbert.utils.bb_splitter import split_into_bbs

from bbert.data.instruction import Vocabulary, InstructionMapping
from bbert.data.dataset import MalwareDataset
from bbert.model.bbert import BBERT

class Inference:
    def __init__(self):
        # self.bbl = BBWModel('bb_stat.npy')
        self.bbl = BBWModel('')

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

    imap = InstructionMapping()
    vmap = Vocabulary(imap)

    model = BBERT(vmap).cuda()
    model.load_state_dict(torch.load('bbert.pth'))
    model = model.bert
    model.eval()

    sep_id = vmap.get_index('[SEP]')
    cls_id = vmap.get_index('[CLS]')

    bert_bbs = []

    for idx, bb in enumerate(bbs):
        bb = [vmap.get_index(inst) for inst in bb]
        bb = torch.cat([
            torch.tensor([cls_id]),
            torch.tensor(bb).long().contiguous(),
            torch.tensor([sep_id])
        ]).long().contiguous()
        bb = bb.unsqueeze(0).cuda()
        bb = model(bb)
        bb = bb.cpu().numpy()[0, 0, :]
        bert_bbs.append(bb)

    inference.analyze(bert_bbs)


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
