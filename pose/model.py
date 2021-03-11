import torch
from einops import repeat
from pose_format.torch.masked import MaskedTorch
from vit_pytorch.vit_pytorch import Transformer
import torch.nn.functional as F

from base_model import PLModule
from pose.args import args, POSE_REP
from pytorch_revgrad import RevGrad

pose_dim = POSE_REP.calc_output_size()


class PoseSequenceClassification(PLModule):
    def __init__(self, dim=512, input_dim=pose_dim, num_classes=226, num_signers=50):
        super().__init__(sign_loss=args.sign_loss, signer_loss=args.signer_loss, signer_loss_patience=args.signer_loss_patience)

        self.batch_norm = torch.nn.BatchNorm1d(num_features=input_dim)
        self.dropout = torch.nn.Dropout(p=0.2)

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

        if args.encoder == "lstm":
            self.encoder = torch.nn.LSTM(input_size=dim, hidden_size=dim//2, num_layers=depth, batch_first=True,
                                         dropout=0.1,
                                         bidirectional=True)
        else:
            self.encoder = Transformer(
                dim=dim,
                depth=depth,
                heads=heads,
                dim_head=dim // heads,
                mlp_dim=dim,
                dropout=0.4
            )

        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = torch.nn.Parameter(torch.randn(1, args.max_seq_size + 1, dim))

        self.head_norm = torch.nn.LayerNorm(dim)
        self.mlp_head = torch.nn.Linear(dim, num_classes)
        self.mlp_signer = torch.nn.Sequential(
            RevGrad(),
            torch.nn.Linear(dim, num_signers)
        )

    def rep_input(self, pose):
        pose = MaskedTorch.squeeze(pose)
        x = POSE_REP(pose).type(torch.float32)
        # print("rep_input", pose.shape,             POSE_REP.calc_output_size(), x.shape)


        return x.squeeze()

    def norm(self, x):
        x = x.transpose(1, 2)
        x = self.batch_norm(x)
        return x.transpose(1, 2)

    def transform(self, x, mask=None):
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        if isinstance(self.encoder, torch.nn.LSTM):
            rep, _ = self.encoder(x)
            input_max, input_indexes = torch.max(rep, dim=1) # max pooling
            return input_max

            # return  torch.mean(rep, dim=1) # avg pooling

        rep = self.encoder(x, mask)
        return rep[:, 0]  # Get CLS token

    def forward(self, batch):
        x = self.rep_input(batch["pose"])
        x = self.dropout(x)
        x = self.norm(x)
        x = self.proj(x)
        x = self.head_norm(self.transform(x, mask=batch["length"]))
        return self.mlp_head(x), self.mlp_signer(x)


if __name__ == "__main__":
    pose = torch.randn(2, 32, 1, POSE_POINTS, POSE_DIMS)
    model = PoseSequenceClassification()
    pred = model(None, None, pose)
    print(pred.shape)
