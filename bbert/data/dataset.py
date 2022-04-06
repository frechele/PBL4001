import glob
import os
import pickle

import torch
from torch.utils.data import Dataset

from bbert.data.instruction import InstructionMapping


class MalwareDataset(Dataset):
    def __init__(self, root_path: str, imap: InstructionMapping):
        super(MalwareDataset, self).__init__()

        self.file_list = glob.glob(os.path.join(root_path, 'pkl', '*.pkl'))
        self.imap = imap

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index: int):
        with open(self.file_list[index], 'rb') as f:
            data = pickle.load(f)

        file_type = data['type']
        bbs = data['bbs']

        bbs = [torch.tensor(list(map(self.imap.get_index, bb))) for bb in bbs]

        return bbs, torch.LongTensor(file_type)


if __name__ == '__main__':
    imap = InstructionMapping()
    dataset = MalwareDataset('data', imap)

    print(len(dataset))
    print(dataset[0])
