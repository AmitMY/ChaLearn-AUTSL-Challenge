from zipfile import ZipFile

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from tqdm import tqdm

# noinspection PyUnresolvedReferences
from pose.data import PoseClassificationDataset, split_train_dataset, get_autsl
from pose.model import PoseSequenceClassification

wandb_logger = WandbLogger(project="autsl", log_model=True)

train, test = get_autsl()
train, val = split_train_dataset(train, {0, 12, 21, 22})

gpus = 1
batch_size = max(1, gpus) * 1024
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val, batch_size=batch_size)

model = PoseSequenceClassification()
# model = PoseSequenceClassification.load_from_checkpoint(
#   "/home/nlp/amit/sign-language/sign-language-recognition/wandb/run-20210131_112537-3pishkjj/files/autsl/3pishkjj/checkpoints/epoch=41-step=1904.ckpt")

trainer = pl.Trainer(
  max_epochs=100,
  logger=wandb_logger,
  gpus=gpus)

# trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)
trainer.test(model, val_loader)

#
#
#
#
#
#
#
#

test_loader = DataLoader(test, batch_size=batch_size)

with open("predictions.csv", "w") as f:
  with torch.no_grad():
    for batch in tqdm(test_loader):
      ids, signers, xs, ys = batch
      signers, xs = signers.cuda(), xs.cuda()

      y_hats = model.pred(ids, signers, xs).cpu().numpy()

      for _id, y_hat in zip(ids, y_hats):
        f.write(_id.decode('utf-8') + "," + str(y_hat) + "\n")

with ZipFile('predictions.zip', 'w') as zipObj:
  zipObj.write("predictions.csv")
