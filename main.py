import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import WandbLogger

import wandb

from bbert.engine.bbert import BBERTModule


if __name__ == '__main__':
    model = BBERTModule()

    model_callback = plc.ModelCheckpoint()

    wandb_logger = WandbLogger(project='bbert', log_model='all')
    wandb_logger.watch(model)

    trainer = pl.Trainer(gpus=1, precision=16, callbacks=[model_callback], max_epochs=10, logger=wandb_logger)
    trainer.fit(model)

    wandb.finish()