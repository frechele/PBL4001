import glob
import os
import pickle
import random

import torch
from torch.utils.data import Dataset

from bbert.data.instruction import InstructionMapping, Vocabulary


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


class BBERTDataset(Dataset):
    def __init__(self, root_path: str, vmap: Vocabulary, seq_len: int=64, mask_frac: float=0.15, random_pick: bool=False):
        super(BBERTDataset, self).__init__()

        self.vmap = vmap

        self.seq_len = seq_len
        self.mask_frac = mask_frac

        self.sep_id = vmap.get_index('[SEP]')
        self.cls_id = vmap.get_index('[CLS]')
        self.pad_id = vmap.get_index('[PAD]')
        self.mask_id = vmap.get_index('[MASK]')

        file_list = glob.glob(os.path.join(root_path, 'pkl', '*.pkl'))

        if os.path.exists('bbert_dataset_cache.pkl'):
            with open('bbert_dataset_cache.pkl', 'rb') as f:
                self.cache, self.access_method = pickle.load(f)
        else:
            self.cache = dict()
            self.access_method = []
            for filename in file_list:
                with open(filename, 'rb') as f:
                    data = pickle.load(f)

                if random_pick:
                    for _ in range(10):
                        self.access_method.append((filename, -1))
                else:
                    self.access_method.extend([(filename, i) for i in range(len(data['bbs']))])
                    
                self.cache[filename] = data

    def __len__(self) -> int:
        return len(self.access_method)

    def __getitem__(self, index: int):
        filename, bb_idx = self.access_method[index]
        data = self.cache[filename]

        if bb_idx < 0:
            bb_idx = random.randint(0, len(data['bbs']) - 1)

        file_type = data['type']
        file_type = torch.tensor(file_type).long()

        bb = [self.vmap.get_index(inst) for inst in data['bbs'][bb_idx]]
        if len(bb) >= self.seq_len - 2:
            bb = bb[:self.seq_len - 2]

        mlm_target = [self.cls_id] + bb + [self.sep_id] + [self.pad_id] * (self.seq_len - 2 - len(bb))
        mlm_target = torch.tensor(mlm_target).long().contiguous()

        def masking(data):
            data = torch.tensor(data).long().contiguous()
            data_len = data.size(0)
            ones_num = int(data_len * self.mask_frac)
            zeros_num = data_len - ones_num
            lm_mask = torch.cat([torch.zeros(zeros_num), torch.ones(ones_num)])
            lm_mask = lm_mask[torch.randperm(data_len)]
            data = data.masked_fill(lm_mask.bool(), self.mask_id)

            return data

        mlm_train = torch.cat([
            torch.tensor([self.cls_id]),
            masking(bb),
            torch.tensor([self.sep_id]),
            torch.tensor([self.pad_id] * (self.seq_len - 2 - len(bb)))
        ]).long().contiguous()

        return mlm_train, mlm_target, file_type


if __name__ == '__main__':
    imap = InstructionMapping()
    vmap = Vocabulary(imap)
    dataset = BBERTDataset('data', vmap)

    print(len(dataset))
    print(dataset[0])
