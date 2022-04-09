from bbert.data.dataset import BBERTDataset
from bbert.data.instruction import Vocabulary, InstructionMapping
from bbert.model.bbert import BBERT

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset

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
        return BBERTDataset('/data/pbl/data', self.vmap, random_pick=False)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, 128, True, num_workers=4, pin_memory=True)

    def on_train_epoch_start(self):
        self.train_acc = 0

    def training_step(self, batch, batch_idx):
        mlm_train, mlm_target, file_type = batch

        pred_mlm, pred_file_type = self.model(mlm_train)

        loss_mlm = F.cross_entropy(pred_mlm.transpose(1, 2), mlm_target, ignore_index=self.pad_id)
        loss_file_type = F.cross_entropy(pred_file_type, file_type)
        
        loss = loss_mlm + loss_file_type

        with torch.no_grad():
            self.train_acc += (pred_file_type.argmax(dim=-1) == file_type).float().mean().item()

        self.log('train_loss_mlm', loss_mlm.item(), prog_bar=True)
        self.log('train_loss_ft', loss_file_type.item(), prog_bar=True)
        self.log('train_acc', self.train_acc / (batch_idx + 1), prog_bar=True)

        return loss

    def configure_optimizers(self):
        opt = optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=2e-5)
        sch = CosineAnnealingWarmRestarts(opt, T_0=5, eta_min=1e-10, last_epoch=-1, T_mult=2)
        return [opt], [sch]
