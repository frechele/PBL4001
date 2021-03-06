import copy
import os
import pickle
from tqdm import tqdm
import multiprocessing as mp
from glob import glob
import numpy as np

JUMP_INSTS = {
    'jo',
    'jno',
    'js',
    'jns',
    'je',
    'jz',
    'jne',
    'jze',
    'jb',
    'jnae',
    'jc',
    'jnb',
    'jae',
    'jnc',
    'jbe',
    'jna',
    'ja',
    'jnbe',
    'jl',
    'jnge',
    'jge',
    'jnl',
    'jle',
    'jng',
    'jg',
    'jnle',
    'jp',
    'jpe',
    'jnp',
    'jpo',
    'jcxz',
    'jecxz',
    'call',
    'ret',
    'retf',
    'retfw',
    'retn',
    'retnw'
}

def convert_one(filename: str):
    basename = os.path.basename(filename)
    basename = os.path.splitext(basename)[0]

    file_type = 1 if '/1/' in filename else 0

    bbs = []
    bb = []

    with open(filename, 'rt') as f:
        lines = f.readlines()

        if len(lines) <= 100:
            return # just skip null or wrong file

        for line in lines:
            line = line.lower().strip()

            bb.append(line)
            if line in JUMP_INSTS:
                bbs.append(bb)
                bb = []

    if len(bb) > 0:
        bbs.append(bb)

    with open(os.path.join('data/pkl/{}.pkl'.format(basename)), 'wb') as f:
        data = {
            'type': file_type,
            'bbs': bbs
        }

        pickle.dump(data, f)


def task(rank: int, file_list):
    total_size = len(file_list)

    for idx, file in enumerate(file_list):
        if rank == 0:
            print('{}/{} {}'.format(idx, total_size, file), flush=True)

        convert_one(file)


if __name__ == '__main__':
    file_list = glob('./data/*/*.txt')

    NUM_PROCS = 28

    procs = []
    length = int(np.ceil(len(file_list) / NUM_PROCS))
    for rank in range(NUM_PROCS):
        list_per_worker = copy.deepcopy(file_list[rank * length:(rank + 1) * length])
        proc = mp.Process(target=task, args=(rank, list_per_worker))
        procs.append(proc)
        proc.start()

    for p in procs:
        p.join()
