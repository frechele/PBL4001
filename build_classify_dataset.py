import glob
import os
import pickle
import time

import torch

from bbert.data.instruction import Vocabulary, InstructionMapping
from bbert.data.dataset import MalwareDataset
from bbert.model.bbert import BBERT


def dataloader(file_list, vmap: Vocabulary):
    sep_id = vmap.get_index('[SEP]')
    cls_id = vmap.get_index('[CLS]')

    for filename in file_list:
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        file_type = data['type']

        for idx, bb in enumerate(data['bbs']):
            bb = [vmap.get_index(inst) for inst in bb]
            bb = torch.cat([
                torch.tensor([cls_id]),
                torch.tensor(bb).long().contiguous(),
                torch.tensor([sep_id])
            ]).long().contiguous()

            yield filename, file_type, bb, (idx == len(data['bbs']) - 1)


class ETA:
    def __init__(self, total_len: int):
        self.start_time = time.time()
        self.total_len = total_len
        self.done_count = 0

    def __call__(self):
        now = time.time()
        self.done_count += 1

        return (now - self.start_time) / self.done_count * (self.total_len - self.done_count)


@torch.no_grad()
def main():
    imap = InstructionMapping()
    vmap = Vocabulary(imap)

    model = BBERT(vmap).cuda()
    model = model.bert
    model.eval()

    file_list = glob.glob('data/pkl/*.pkl')
    eta = ETA(len(file_list))

    inst_buffer = []
    for filename, file_type, bb, done in dataloader(file_list, vmap):
        bb = bb.unsqueeze(0).cuda()
        bb = model(bb)
        bb = bb.cpu().numpy()
        inst_buffer.append(bb)

        if done:
            with open('data/pkl2/{}'.format(os.path.basename(filename)), 'wb') as f:
                pickle.dump({
                    'type': file_type,
                    'bbs': inst_buffer
                }, f)
            inst_buffer = []

            print('{} ETA: {} seconds'.format(filename, eta()), flush=True)


if __name__ == '__main__':
    main()
