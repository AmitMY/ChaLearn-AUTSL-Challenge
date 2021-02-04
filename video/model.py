import torch
from linformer import Linformer
from vit_pytorch.efficient import ViT

from base_model import PLModule
from .data import IMG_SIZE, CHANNELS


class ViTVideoClassification(PLModule):
  def __init__(self, patch_size=32, dim=128, num_classes=226):
    super().__init__()

    self.norm = torch.nn.BatchNorm2d(num_features=CHANNELS)

    efficient_transformer = Linformer(
      dim=dim,
      seq_len=(IMG_SIZE // patch_size) ** 2 + 1,  # + 1 cls token
      depth=12,
      heads=8,
      k=64
    )

    self.vit = ViT(
      dim=dim,
      image_size=IMG_SIZE,
      patch_size=patch_size,
      num_classes=num_classes,
      channels=CHANNELS,
      transformer=efficient_transformer
    )

  def forward(self, x):
    x = self.norm(x)
    return self.vit(x)


if __name__ == "__main__":
  img = torch.randn(2, 3, 256, 256)
  model = ViTVideoClassification()
  pred = model(img)
  print(pred.shape)
