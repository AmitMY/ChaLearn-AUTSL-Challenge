import torch
from einops import repeat
from pose_format.torch.masked import MaskedTorch

from base_model import PLModule
from args import args, POSE_REP
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

        if args.encoder == "lstm":
            self.encoder = torch.nn.LSTM(input_size=dim, hidden_size=dim // 2, num_layers=depth, batch_first=True,
                                         dropout=0.1,
                                         bidirectional=True)
        else:
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dim * 4,  # Typically this is set to 4*dim
                dropout=0.4,
                activation='gelu'  # or 'relu' based on your preference
            )
            self.encoder = torch.nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=depth
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
        return x.squeeze()

    def norm(self, x):
        x = x.transpose(1, 2)
        x = self.batch_norm(x)
        return x.transpose(1, 2)

    def transform(self, x, mask=None):
        #print(f"x_inputed ({x.shape})") # x_inputed (torch.Size([512, 116, 512]))
        b, n, _ = x.shape

        mask = mask.to(x.device) if mask is not None else None


        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        if isinstance(self.encoder, torch.nn.LSTM):
            rep, _ = self.encoder(x)
            input_max, input_indexes = torch.max(rep, dim=1)  # max pooling
            print(f"rep ({rep.shape})")
            print(f"input_max ({input_max.shape})")
            print(f"input_indexes ({input_indexes.shape})")
            return input_max

        #print(f"mask_before ({mask.shape}): \n {mask}") # mask_before (torch.Size([512, 116])):

        # Invert the mask to match PyTorch's expectation and add padding for the CLS token
        if mask is not None:
            mask = ~mask
            # Add a False value at the beginning of each sequence in the mask for the CLS token
            mask = torch.cat((torch.zeros(mask.size(0), 1, device=mask.device, dtype=mask.dtype), mask), dim=1)

            #mask = mask.transpose(0, 1)

        #print(f"mask_after ({mask.shape}): \n {mask}") # mask_after (torch.Size([117, 512])):
        #print(f"x_after ({x.shape})") # x_after (torch.Size([512, 117, 512])) (B, T, E)
        x = x.transpose(0, 1) # (B, T, E) -> (T, B, E)

        x = self.encoder(x, src_key_padding_mask=mask).transpose(0, 1) # (T, B, E) - >(B, T, E)

        #print(f"x_final ({x.shape})") # x_after (torch.Size([512, 117, 512])) (B, T, E)
        return x[:, 0]  # Get CLS token

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
