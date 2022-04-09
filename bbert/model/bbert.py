from bbert.data.instruction import Vocabulary
from bbert.model.bert import BERT

import torch
import torch.nn as nn
from typing import Tuple


class BBERT(nn.Module):
    def __init__(self, vmap: Vocabulary):
        super(BBERT, self).__init__()

        self.bert = BERT(vmap)

        self.mlm = nn.Linear(self.bert.hidden, vmap.size - 1)
        self.malware_classifier = nn.Linear(self.bert.hidden, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.bert(x)

        return self.mlm(x), self.malware_classifier(x[:, 0])
