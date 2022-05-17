import numpy as np
from typing import List
from scipy.stats import norm


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
