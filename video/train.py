import random

import cv2
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import get_autsl, GRID_SIZE
from model import ViTVideoClassification

# model = model.cuda()
# model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
#
# img = torch.tensor([get_mock_datum()] * 20, dtype=torch.float32)  # high resolution picture
# img = img.cuda()
# for i in tqdm(range(1000)):
#     model(img)  # (1, NUM_CLASSES)

wandb_logger = WandbLogger(project="autsl", log_model=True, offline=True)

# dataset = get_mock_dataset(num_samples=1000)
# train, val = random_split(dataset, [900, 100])
train, val, test = get_autsl()

img = (random.choice(train.data)[1][0].numpy().T + 0.5) * 256
print("img", img.shape)
cv2.imwrite("test.png", img)

gpus = 1
batch_size = max(1, gpus) * 800 // (GRID_SIZE ** 2)
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val, batch_size=batch_size)

# 32 /home/nlp/amit/sign-language/sign-language-recognition/wandb/run-20210112_172625-akenwt7g/files/autsl/akenwt7g/checkpoints/epoch=99-step=3699.ckpt


# model = ViTVideoClassification()
model = ViTVideoClassification.load_from_checkpoint(
    "/home/nlp/amit/sign-language/sign-language-recognition/wandb/run-20210125_115451-27k27yik/files/autsl/27k27yik/checkpoints/epoch=99-step=1199.ckpt")

trainer = pl.Trainer(
    max_epochs=100,
    logger=wandb_logger,
    gpus=gpus)

# trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)
trainer.test(model, val_loader)
#
#
# # Predict on test
# print("\n" * 5)
# print("Predicting on test")
# predictions = {}
# with torch.no_grad():
#     for i in tqdm(range(0, len(test.data), batch_size)):
#         batch = test.data[i: i + batch_size]
#         _ids = [d[0].decode("utf-8") for d in batch]
#         xs = torch.stack([d[1][0] for d in batch], 0)
#         ys = torch.stack([d[1][1] for d in batch], 0)
#         y_hats = model.pred(xs)
#         print(torch.sum(ys == y_hats) / len(ys))
#         y_hats = y_hats.cpu().numpy().tolist()
#         for _id, y in zip(_ids, y_hats):
#             predictions[_id] = y
#
# # rows = []
# # pred_order = open("predictions_order.csv", "r").read().strip().splitlines()
# # for row in pred_order:
# #     _id, _ = row.split(",")
# #     if _id in predictions:
# #         rows.append(_id + "," + str(predictions[_id]))
# #
# # with open("predictions.csv", "w") as f:
# #     f.write("\n".join(rows))
