import numpy as np
from typing import List
from scipy.stats import norm
from sklearn.svm import SVC

import torch
import os
import pickle
import glob

from bbert.data.instruction import Vocabulary, InstructionMapping
from bbert.data.dataset import MalwareDataset
from bbert.model.bbert import BBERT

DATASET_PATH = '/data/pbl/data/pkl2'
DATA_LIST = glob.glob(os.path.join(DATASET_PATH, '*.pkl'))

MU = 0.5
SIGMA = 1

class BBLModel:
    def __init__(self, ckpt_filename: str):
        ckpt = np.load(ckpt_filename)
        self.std_mu, self.std_std = ckpt[0]
        self.mean_mu, self.mean_std = ckpt[1]

    def calc_score(self, bbs: List[str], weight: float=0.5) -> float:
        def zscore(x, mu, sigma):
            return (x - mu) / sigma

        bb_lengths = np.array([len(bb) for bb in bbs])
        bb_mean, bb_std = bb_lengths.mean(), bb_lengths.std()

        std_z = zscore(bb_std, self.std_mu, self.std_std)
        mean_z = zscore(bb_mean, self.mean_mu, self.mean_std)

        std_score = 2 * (1 - norm.cdf(np.abs(std_z)))
        mean_score = 2 * (1 - norm.cdf(np.abs(mean_z)))

        return 1 - (std_score ** 0.44404637) * (mean_score ** 0.33140396)

class BBWModel:
    def __init__(self, ckpt_filename: str):
        if ckpt_filename != "":
            ckpt = np.load(ckpt_filename)
            self.mu, self.sigma = ckpt
        else :
            self.mu = MU
            self.sigma = SIGMA
        self.predict_model = self.bulid_predict_model()

    def _get_weights(self ,bbs_length):
        if bbs_length > 1:
            idxs = np.arange(bbs_length) / (bbs_length - 1)
        else:
            idxs = np.array([1.])

        # weights = norm.pdf(idxs, self.mu, self.sigma)
        weights = np.ones((bbs_length))
        weights_sum = np.sum(weights)
        
        return weights , weights_sum

    def calc_score(self, bbs: List[str]):

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

        total_length = len(bert_bbs)
        weights , weights_sum = self._get_weights(total_length)

        weighted_bbs = weights * np.array(bert_bbs) / weights_sum
        return self.predict_model.predict(weighted_bbs.tolist())
    
    def bulid_predict_model(self):
        bb_types = np.zeros(len(DATA_LIST))
        bbs = np.zeros((len(DATA_LIST), 768))

        for i, filename in enumerate(DATA_LIST):
            filetype, pe = self.get_program_encoding(filename)

            bb_types[i] = filetype
            bbs[i] = pe

        return SVC(kernel='linear').fit(bbs, bb_types)

    def get_program_encoding(self, filename: str):
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        filetype = data['type']
        total_length = len(data['bbs'])

        weights, weights_sum = self._get_weights(total_length)
        bb = np.array(data['bbs'])
        pe = np.sum(bb * weights[..., np.newaxis], axis=0)

        return filetype, pe / weights_sum