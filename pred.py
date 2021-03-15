from collections import Counter
from zipfile import ZipFile

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .args import args
from .data import get_autsl, ZeroPadCollator
from .model import PoseSequenceClassification


model = PoseSequenceClassification.load_from_checkpoint(
  "/home/nlp/amit/sign-language/sign-language-recognition/pose/models/2glnyy2o/weights.ckpt.ckpt")
model = model.cuda()


test = get_autsl('validation')
test.is_train = False

collator = ZeroPadCollator()
test_loader = DataLoader(test, batch_size=args.batch_size, collate_fn=collator.collate)

predictions = {}
gold_values = {datum["id"]: datum["label"] for datum in test.data}

with torch.no_grad():
  for batch in tqdm(test_loader):
    batch["pose"] = batch["pose"].cuda()

    y_hats = model.pred(batch).cpu().numpy()

    for _id, y_hat in zip(batch["id"], y_hats):
      predictions[_id] = y_hat


correct = len([1 for _id, y_hat in predictions.items() if y_hat == gold_values[_id]])
print("Accuracy", correct / len(predictions))

for _id, y_hat in predictions.items():
  if y_hat == gold_values[_id]:
    print(y_hat, gold_values[_id])

with open("predictions.csv", "w") as f:
  for _id, y_hat in predictions.items():
    f.write(_id.decode('utf-8') + "," + str(y_hat) + "\n")

with ZipFile('predictions.zip', 'w') as zipObj:
  zipObj.write("predictions.csv")
