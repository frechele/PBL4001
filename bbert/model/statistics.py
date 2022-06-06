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
    def __init__(self, bbert_filename:str = 'bbert.pth' , pdmd_filename:str = 'pdmd.pkl'):
        self.predict_model = pickle.loads(pdmd_filename)
        
        self.imap = InstructionMapping()
        self.vmap = Vocabulary(self.imap)
        
        self.model = BBERT(self.vmap).cuda()
        self.model.load_state_dict(torch.load(bbert_filename))
        self.model = self.model.bert
        self.model.eval()

        self.sep_id = self.vmap.get_index('[SEP]')
        self.cls_id = self.vmap.get_index('[CLS]')

    def _get_weights(self ,bbs_length):
        weights = np.ones((bbs_length))
        weights_sum = np.sum(weights)
        
        return weights, weights_sum

    def calc_score(self, bbs: List[str]):
        bert_bbs = []

        for idx, bb in enumerate(bbs):
            bb = [self.vmap.get_index(inst) for inst in bb]
            bb = torch.cat([
                torch.tensor([self.cls_id]),
                torch.tensor(bb).long().contiguous(),
                torch.tensor([self.sep_id])
            ]).long().contiguous()

            bb = bb.unsqueeze(0).cuda()
            bb = self.model(bb)
            bb = bb.cpu().numpy()[0, 0, :]
            bert_bbs.append(bb)

        total_length = len(bert_bbs)
        weights, weights_sum = self._get_weights(total_length)
        weighted_bbs = weights * np.array(bert_bbs) / weights_sum

        return self.predict_model.predict(weighted_bbs)