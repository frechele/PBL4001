from bbert.data.dataset import BBERTDataset
from bbert.data.instruction import Vocabulary, InstructionMapping
from bbert.model.bbert import BBERT

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset
import madgrad

import pytorch_lightning as pl


class BBERTModule(pl.LightningModule):
    def __init__(self):
        super(BBERTModule, self).__init__()

        self.imap = InstructionMapping()
        self.vmap = Vocabulary(self.imap)

        self.pad_id = self.vmap.get_index('[PAD]')

        self.model = BBERT(self.vmap)

    @property
    def train_dataset(self) -> Dataset:
        return BBERTDataset('/data/pbl/data', self.vmap, random_pick=True)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, 256, True, num_workers=4, pin_memory=True)

    def training_step(self, batch, batch_idx):
        mlm_train, mlm_target, file_type = batch

        _, pred_mlm = self.model(mlm_train)

        loss_mlm = F.cross_entropy(pred_mlm.transpose(1, 2), mlm_target, ignore_index=self.pad_id)
        
        loss = loss_mlm

        self.log('train_loss_mlm', loss_mlm.item(), prog_bar=True)

        return loss

    def configure_optimizers(self):
        opt = madgrad.MADGRAD(self.model.parameters(), lr=1e-5)
        sch = CosineAnnealingWarmRestarts(opt, T_0=5, eta_min=1e-10, last_epoch=-1, T_mult=2)
        return [opt], [sch]
