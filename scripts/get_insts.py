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

    with open('instruction_set.pkl', 'wb') as f:
        pickle.dump(inst_set, f)

    print(inst_set)

