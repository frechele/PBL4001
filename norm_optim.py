import ray
import pickle
import os
import numpy as np
from collections import namedtuple
import glob
from scipy.stats import norm
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import random
import tqdm


################################
# Configurations               #
################################
BIG_BANK = 100
N_BANK = 50
SAMPLE_SIZE = 200
TOTAL_ITER = 100
################################
# Configurations               #
################################

DATASET_PATH = '/data/pbl/data/pkl3'
NORMAL_LIST = glob.glob(os.path.join(DATASET_PATH, '0', '*.pkl'))
MALWARE_LIST = glob.glob(os.path.join(DATASET_PATH, '1', '*.pkl'))

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

        weights = gaussian(idxs, seed)
        weight_sum = np.sum(weights)
        bb = np.array(data['bbs'])
        pe = np.sum(bb * weights[..., np.newaxis], axis=0)
        pe = pe / weight_sum
    else:
        pe = np.array(data['bbs'])
    
    return filetype, pe


@ray.remote
def calc_score(seed: Seed):
    samples = random.sample(NORMAL_LIST, SAMPLE_SIZE//2) + random.sample(MALWARE_LIST, SAMPLE_SIZE//2)

    bb_types = np.zeros(SAMPLE_SIZE)
    bbs = np.zeros((SAMPLE_SIZE, 768))

    for i, filename in enumerate(samples):
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


def print_top5_bank(bank):
    for i, (seed, score) in enumerate(bank[:5], 1):
        print('top{}'.format(i), seed, score)


if __name__ == '__main__':
    ray.init(num_cpus=28, dashboard_port=40000, dashboard_host='0.0.0.0')

    if not os.path.exists('firstbank.pkl'):
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

        with open('firstbank.pkl', 'wb') as f:
            pickle.dump(bank, f)
    else:
        print('load first bank checkpoint')
        with open('firstbank.pkl', 'rb') as f:
            bank = pickle.load(f)

    print()
    print('<first bank checkpoint>')
    print_top5_bank(bank)

    for it in range(TOTAL_ITER):
        print(f'iteration {it}/{TOTAL_ITER}', flush=True)

        bank_candidates = []
        for _ in range(BIG_BANK - N_BANK):
            if np.random.random() < 0.5:
                seed1, seed2 = map(lambda x: x[0], random.sample(bank, 2))
                bank_candidates.append(crossover(seed1, seed2))
            else:
                seed = random.choice(bank)[0]
                bank_candidates.append(mutation(seed))

        bank += distribute(bank_candidates)
        bank = sorted(bank, key=lambda x: x[1], reverse=True)
        bank = bank[:N_BANK]

        print('checkpoint')
        print_top5_bank(bank)
        print(flush=True)

        with open(f'iteration{it:04d}.pkl', 'wb') as f:
            pickle.dump(bank, f)

    ray.shutdown()
