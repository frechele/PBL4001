import ray
import pickle
import os
import numpy as np
from collections import namedtuple
import glob
from scipy.stats import norm
from sklearn.svm import SVC
from sklearn.metrics import f1_score


DATASET_PATH = '/data/pbl/data/pkl2'
DATA_LIST = glob.glob(os.path.join(DATASET_PATH, '*.pkl'))

Seed = namedtuple('Seed', ['mu', 'std'])


def crossover(seed1: Seed, seed2: Seed) -> Seed:
    w1 = np.random.uniform(0, 1)
    w2 = np.random.uniform(0, 1)

    new_mu = w1 * seed1.mu + (1 - w1) * seed2.mu
    new_std = w2 * seed1.std + (1 - w2) * seed2.mu

    return Seed(mu=new_mu, std=new_std)


def mutation(seed: Seed) -> Seed:
    w1 = np.random.uniform(-1, 1) * 0.01
    w2 = np.random.uniform(-1, 1) * 0.01

    new_mu = np.clip(seed.mu + w1, 0, 1)
    new_std = np.clip(seed.std + w2, 0, 1)
    
    return Seed(mu=new_mu, std=new_std)


def gaussian(x, seed: Seed) -> float:
    return norm.pdf(x, loc=seed.mu, scale=seed.std)


def get_program_encoding(filename: str, seed: Seed):
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    filetype = data['type']
    total_length = len(data['bbs'])
    if total_length > 1:
        idxs = np.arange(total_length) / (total_length - 1)
    else:
        idxs = np.array([1.])

    weights = gaussian(idxs, seed)
    weight_sum = np.sum(weights)
    bb = np.array(data['bbs'])
    pe = np.sum(bb * weights[..., np.newaxis], axis=0)
    
    return filetype, pe / weight_sum


BIG_BANK = 100
N_BANK = 50


@ray.remote
def calc_score(seed: Seed):
    bb_types = np.zeros(len(DATA_LIST))
    bbs = np.zeros((len(DATA_LIST), 768))

    for i, filename in enumerate(DATA_LIST):
        filetype, pe = get_program_encoding(filename, seed)

        bb_types[i] = filetype
        bbs[i] = pe

    model = SVC(kernel='linear').fit(bbs, bb_types)
    preds = model.predict(bbs)

    return seed, f1_score(bb_types, preds)


def distribute(seeds: list):
    futures = []
    for seed in seeds:
        seed = ray.put(seed)
        futures.append(calc_score.remote(seed))
    return ray.get(futures)


if __name__ == '__main__':
    ray.init(num_cpus=28, dashboard_port=40000, dashboard_host='0.0.0.0')

    print('build the first bank', end=' ')
    big_bank = []
    for _ in range(BIG_BANK):
        mu = np.random.random()
        std = np.random.random()
        big_bank.append(Seed(mu, std))

    bank = distribute(big_bank)
    bank = sorted(bank, key=lambda x: x[1], reverse=True)
    bank = bank[:N_BANK]
    print('DONE')

    ray.shutdown()
