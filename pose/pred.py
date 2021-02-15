from collections import Counter
from zipfile import ZipFile

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pose.args import args
from pose.data import get_autsl
from pose.model import PoseSequenceClassification

test = get_autsl('validation')

model = PoseSequenceClassification.load_from_checkpoint(
  "/home/nlp/amit/sign-language/sign-language-recognition/pose/wandb/run-20210206_170707-111wpn4k/files/autsl/111wpn4k/checkpoints/epoch=26-step=606.ckpt")
model = model.cuda()

test.is_train = True  # TODO try majority voting for validation set, by doing augmentation?
pred_runs = 1
# test.is_train = False
# pred_runs = 1

test_loader = DataLoader(test, batch_size=args.batch_size)

predictions = {datum["id"]: [] for datum in test.data}
gold_values = {datum["id"]: datum["label"] for datum in test.data}

with torch.no_grad():
  for i in range(pred_runs):
    for batch in tqdm(test_loader):
      ids, signers, xs, ys = batch
      signers, xs = signers.cuda(), xs.cuda()

      y_hats = model.pred(ids, signers, xs).cpu().numpy()

      for _id, y_hat in zip(ids, y_hats):
        predictions[_id].append(y_hat)

most_common_pred = {_id: Counter(y_hats).most_common(1)[0][0] for _id, y_hats in predictions.items()}

correct = len([1 for _id, y_hat in most_common_pred.items() if y_hat == gold_values[_id]])
print("Accuracy", correct / len(gold_values))

with open("predictions.csv", "w") as f:
  for _id, y_hat in most_common_pred.items():
    f.write(_id.decode('utf-8') + "," + str(y_hat) + "\n")

with ZipFile('predictions.zip', 'w') as zipObj:
  zipObj.write("predictions.csv")