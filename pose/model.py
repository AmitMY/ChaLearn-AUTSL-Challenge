import torch
from einops import repeat
from linformer import Linformer

from base_model import PLModule
from pose.data import POSE_POINTS, POSE_DIMS


class PoseSequenceClassification(PLModule):
  def __init__(self, seq_len=32, dim=256, input_dim=POSE_POINTS * POSE_DIMS, num_classes=226):
    super().__init__()

    self.norm = torch.nn.BatchNorm1d(num_features=input_dim)

    self.proj = torch.nn.Linear(in_features=input_dim, out_features=dim)

    self.transformer = Linformer(
      dim=dim,
      seq_len=seq_len + 1,  # + 1 cls token
      depth=2,
      heads=4,
      k=64,
      dropout=0.4
    )

    self.cls_token = torch.nn.Parameter(torch.randn(1, 1, dim))
    self.pos_embedding = torch.nn.Parameter(torch.randn(1, seq_len + 1, dim))

    self.mlp_head = torch.nn.Sequential(
      torch.nn.LayerNorm(dim),
      torch.nn.Linear(dim, num_classes)
    )

  def norm_input(self, x):
    batch, seq_len, people, points, dim = x.shape
    x = x.view(batch, seq_len, -1)
    # return x

    x = x.transpose(1, 2)
    x = self.norm(x)
    return x.transpose(1, 2)

  def transform(self, x):
    b, n, _ = x.shape

    cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
    x = torch.cat((cls_tokens, x), dim=1)
    x += self.pos_embedding[:, :(n + 1)]

    return self.transformer(x)

  def forward(self, _id, signer, x):
    x = self.norm_input(x)
    x = self.proj(x)
    x = self.transform(x)

    x = x[:, 0]  # Get CLS token
    return self.mlp_head(x)


if __name__ == "__main__":
  pose = torch.randn(2, 32, 1, POSE_POINTS, POSE_DIMS)
  model = PoseSequenceClassification()
  pred = model(None, None, pose)
  print(pred.shape)
