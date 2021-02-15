import torch
from einops import repeat
from linformer import Linformer
from vit_pytorch.vit_pytorch import Transformer
from pose_format.torch.masked import MaskedTensor

from base_model import PLModule
from pose.args import POSE_REP, args

pose_dim = POSE_REP.calc_output_size()


class PoseSequenceClassification(PLModule):
  def __init__(self, seq_len=32, dim=512, input_dim=pose_dim, num_classes=226):
    super().__init__()

    self.batch_norm = torch.nn.BatchNorm1d(num_features=input_dim)

    self.proj = torch.nn.Linear(in_features=input_dim, out_features=dim)

    heads = args.encoder_heads
    depth = args.encoder_depth

    # self.transformer = Linformer(
    #   dim=dim,
    #   seq_len=seq_len + 1,  # + 1 cls token
    #   depth=depth,
    #   heads=heads,
    #   k=64,
    #   dropout=0.4
    # )

    self.transformer = Transformer(
      dim=dim,
      depth=depth,
      heads=heads,
      dim_head=dim // heads,
      mlp_dim=dim,
      dropout=0.4
    )

    self.cls_token = torch.nn.Parameter(torch.randn(1, 1, dim))
    self.pos_embedding = torch.nn.Parameter(torch.randn(1, seq_len + 1, dim))

    self.mlp_head = torch.nn.Sequential(
      torch.nn.LayerNorm(dim),
      torch.nn.Linear(dim, num_classes)
    )

  def rep_input(self, x):
    x = torch.squeeze(x)
    masked = MaskedTensor(x)
    x = POSE_REP(masked)

    return x.squeeze()

  def norm(self, x):
    x = x.transpose(1, 2)
    x = self.batch_norm(x)
    return x.transpose(1, 2)

  def transform(self, x):
    b, n, _ = x.shape

    cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
    x = torch.cat((cls_tokens, x), dim=1)
    x += self.pos_embedding[:, :(n + 1)]

    return self.transformer(x)

  def forward(self, _id, signer, x):
    x = self.rep_input(x)
    x = self.norm(x)
    x = self.proj(x)
    x = self.transform(x)

    x = x[:, 0]  # Get CLS token
    return self.mlp_head(x)


if __name__ == "__main__":
  pose = torch.randn(2, 32, 1, POSE_POINTS, POSE_DIMS)
  model = PoseSequenceClassification()
  pred = model(None, None, pose)
  print(pred.shape)
