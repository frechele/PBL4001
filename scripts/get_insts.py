from glob import glob
import pickle
from tqdm import tqdm


if __name__ == '__main__':
    files = glob('data/*/*.txt')

    inst_set = set()

    for filename in tqdm(files):
        with open(filename, 'rt') as f:
            for line in f.readlines():
                inst_set.add(line.lower().strip())

    inst_set = list(inst_set)

    inst2ind = dict()
    ind2inst = dict()

    for idx, inst in enumerate(inst_set):
        inst2ind[inst] = idx
        ind2inst[idx] = inst

    with open('instruction_set.pkl', 'wb') as f:
        pickle.dump({
            'inst2ind': inst2ind,
            'ind2inst': ind2inst
        }, f)
