from unicodedata import bidirectional
import torch
import torch.nn as nn

from typing import List

from bbert.data.instruction import InstructionMapping


class BasicBlockEncoder(nn.Module):
    EMBED_DIM = 256

    def __init__(self, imap: InstructionMapping):
        super(BasicBlockEncoder, self).__init__()

        self.embed = nn.Embedding(imap.size, BasicBlockEncoder.EMBED_DIM)
        self.lstm = nn.LSTM(input_size=BasicBlockEncoder.EMBED_DIM, hidden_size=128,
                            num_layers=2, dropout=0.3, bidirectional=True, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        x = self.embed(x)

        hidden = [torch.zeros(2 * 2, batch_size, 128, requires_grad=True),
                  torch.zeros(2 * 2, batch_size, 128, requires_grad=True)]

        outputs, _ = self.lstm(x, hidden)
        return outputs[:, -1, :]


class ProgramEncoder(nn.Module):
    EMBED_DIM = 256

    def __init__(self, bb_encoder: BasicBlockEncoder):
        super(ProgramEncoder, self).__init__()

        self.bb_encoder = bb_encoder
        self.lstm = nn.LSTM(input_size=ProgramEncoder.EMBED_DIM, hidden_size=128,
                            num_layers=2, dropout=0.3, bidirectional=True, batch_first=True)

    def forward(self, bbs: List[torch.Tensor]) -> torch.Tensor:
        bb_embeddings = []
        for bb in bbs:
            bb = bb.view(1, -1)
            embed = self.bb_encoder(bb)
            bb_embeddings.append(embed)

        bb_embeddings = torch.cat(bb_embeddings, dim=0).unsqueeze(0)
        
        hidden = [torch.zeros(2 * 2, 1, 128, requires_grad=True),
                  torch.zeros(2 * 2, 1, 128, requires_grad=True)]

        outputs, _ = self.lstm(bb_embeddings, hidden)
        return outputs[:, -1, :]


if __name__ == '__main__':
    from bbert.data.dataset import MalwareDataset

    imap = InstructionMapping()
    dataset = MalwareDataset('data', imap)

    bbs, target = dataset[0]

    bb_net = BasicBlockEncoder(imap)
    net = ProgramEncoder(bb_net)
    
    print(net(bbs))
