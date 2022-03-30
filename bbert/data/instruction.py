import pickle
from typing import Dict, Union

class InstructionMapping:
    def __init__(self, db_file: str='instruction_set.pkl'):
        with open(db_file, 'rb') as f:
            data = pickle.load(f)

        self.inst2ind: Dict[str, int] = data['inst2ind']
        self.ind2inst: Dict[int, str] = data['ind2inst']

    def get_index(self, inst: str) -> Union[int, None]:
        return self.inst2ind.get(inst, None)

    def get_inst(self, idx: int) -> Union[str, None]:
        return self.ind2inst.get(idx, None)

    @property
    def size(self) -> int:
        return len(self.ind2inst)
