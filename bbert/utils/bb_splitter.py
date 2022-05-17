from typing import List


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

def split_into_bbs(instructions: List[str]) -> List[List[str]]:
    bbs = []
    bb = []

    if len(instructions) <= 100:
        return None
    
    for inst in instructions:
        bb.append(inst)
        if inst in JUMP_INSTS:
            bbs.append(bb)
            bb = []

    if len(bb) > 0:
        bbs.append(bb)

    return bbs
