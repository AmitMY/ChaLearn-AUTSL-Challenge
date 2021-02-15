import pickle
import random
from argparse import ArgumentParser
from os import path

import numpy as np
import torch
from pose_format.pose_header import PoseHeader, PoseHeaderComponent
from pose_format.torch.pose_representation import TorchPoseRepresentation
from pose_format.torch.representation.angle import AngleRepresentation
from pose_format.torch.representation.distance import DistanceRepresentation
from pose_format.torch.representation.points import PointsRepresentation
from pose_format.utils.reader import BufferReader

root_dir = path.dirname(path.realpath(__file__))
parser = ArgumentParser()

# Training Arguments
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--gpus', type=int, default=1, help='how many gpus')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--val_signers', type=list, default=[0, 12, 17, 21, 22, 29], help='signers for the validation set')

# Data Arguments
parser.add_argument('--seq_size', type=int, default=32, help='input sequence size')
parser.add_argument('--pose_dims', type=int, default=2, help='x, y, z and k (2, 3, or 4)')
parser.add_argument('--pose_components', type=list,
                    default=["POSE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"],
                    help='what pose components to use?')

# Model Arguments
parser.add_argument('--encoder_depth', type=int, default=4, help='number of layers for the encoder')
parser.add_argument('--encoder_heads', type=int, default=4, help='number of heads for the encoder')

# Augmentation Arguments
parser.add_argument('--rotation_std', type=float, default=0.05, help='augmentation rotation std')
parser.add_argument('--shear_std', type=float, default=0.01, help='augmentation shear std')
parser.add_argument('--scale_std', type=float, default=0.01, help='augmentation scale std')

# Representation Arguments
parser.add_argument('--rep_points', type=bool, default=True, help='use raw points in the vectors?')
parser.add_argument('--rep_distance', type=bool, default=True, help='use limb distance in the vectors?')
parser.add_argument('--rep_angles', type=bool, default=True, help='use limb angle in the vectors?')

# each LightningModule defines arguments relevant to it
args = parser.parse_args()

with open("pose.header", "rb") as f:
  POSE_HEADER = PoseHeader.read(BufferReader(f.read()))

# ---------------------
# Set Representations
# ---------------------
rep_modules1 = []
rep_modules2 = []
if args.rep_points:
  rep_modules1.append(PointsRepresentation())
if args.rep_distance:
  rep_modules2.append(DistanceRepresentation())
if args.rep_angles:
  rep_modules2.append(AngleRepresentation())

components = []
for c in POSE_HEADER.components:
  if c.name in args.pose_components:
    c_copy = pickle.loads(pickle.dumps(c))
    c_copy.format = c_copy.format[:args.pose_dims]
    components.append(c_copy)

rep_header = PoseHeader(version=POSE_HEADER.version, dimensions=POSE_HEADER.dimensions, components=components)
POSE_REP = TorchPoseRepresentation(header=rep_header, rep_modules1=rep_modules1, rep_modules2=rep_modules2)

# ---------------------
# Set Seed
# ---------------------
if args.seed == 0:  # Make seed random if 0
  args.seed = random.randint(0, 1000)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
