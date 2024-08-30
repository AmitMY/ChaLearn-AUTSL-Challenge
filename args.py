import os
import pickle
import random
from argparse import ArgumentParser
from os import path

import numpy as np
import torch
from pose_format.pose_header import PoseHeader
from pose_format.torch.pose_representation import TorchPoseRepresentation
from pose_format.torch.representation.angle import AngleRepresentation
from pose_format.torch.representation.distance import DistanceRepresentation
from pose_format.torch.representation.points import PointsRepresentation
from pose_format.utils.reader import BufferReader

def valid_appearance_option(value):
    if value in ["from_splits_signers", "all_splits_appearances"]:
        return value
    elif os.path.isfile(value):
        return value
    else:
        raise argparse.ArgumentTypeError(f"Invalid value for --appearances: {value}. Must be one of 'from_splits_signers', 'all_splits_appearances', or a valid file path.")

root_dir = path.dirname(path.realpath(__file__))
parser = ArgumentParser()

parser.add_argument('--no_wandb', type=bool, default=False, help='ignore wandb?')
# Training Arguments
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--gpus', type=int, default=1, help='how many gpus')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
# parser.add_argument('--val_signers', type=list, default=[0, 12, 8, 22], help='signers for the validation set')

# Data Arguments
parser.add_argument('--holistic', type=bool, default=True, help='Load holistic?')
parser.add_argument('--openpose', type=bool, default=False, help='Load openpose?')

parser.add_argument('--max_seq_size', type=int, default=512, help='input sequence size')
parser.add_argument('--fps', type=int, default=30, help='fps to load')
parser.add_argument('--pose_dims', type=int, default=3, help='x, y, z and k (2, 3, or 4)')
parser.add_argument('--holistic_pose_components', type=list,
                    default=["POSE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"], # , "FACE_LANDMARKS"
                    help='what pose components to use?')

# Model Arguments
parser.add_argument('--encoder', choices=['lstm', 'transformer'], default='transformer', help='lstm or transformer')
parser.add_argument('--encoder_depth', type=int, default=2, help='number of layers for the encoder')
parser.add_argument('--encoder_heads', type=int, default=4, help='number of heads for the encoder')

parser.add_argument('--sign_loss', type=float, default=1, help='sign loss weight')
parser.add_argument('--signer_loss', type=float, default=0, help='signer loss weight')
parser.add_argument('--signer_loss_patience', type=int, default=0,
                    help='number of epochs before signer loss is applied')

# Augmentation Arguments
parser.add_argument('--frame_dropout_std', type=float, default=0, help='augmentation rotation std')
parser.add_argument('--rotation_std', type=float, default=0, help='augmentation rotation std')
parser.add_argument('--shear_std', type=float, default=0, help='augmentation shear std')
parser.add_argument('--scale_std', type=float, default=0, help='augmentation scale std')

# Representation Arguments
parser.add_argument('--rep_points', type=bool, default=True, help='use raw points in the vectors?')
parser.add_argument('--rep_distance', type=bool, default=True, help='use limb distance in the vectors?')
parser.add_argument('--rep_angles', type=bool, default=True, help='use limb angle in the vectors?')

# Pose Anonymization Arguments
parser.add_argument('--anonymize', type=bool, default=False, help='If True, anonymizes the poses.')
parser.add_argument('--transfer_appearance', type=bool, default=False, help='If True, randomly transfers the apperance of poses.')
parser.add_argument('--appearances', type=valid_appearance_option, default="from_splits_signers", 
                    help="Choose from 'from_splits_signers', 'all_splits_appearances', or provide a valid file path.")

# Add arguments for output paths
parser.add_argument('--log_dir', type=str, default='logs', help='Path in which to save logs')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Path in which to save checkpoints')

# Add arguments for evaluation
parser.add_argument('--pretrained_checkpoint', type=str, default='checkpoints', help='Path to the model to be evaluated.')
parser.add_argument('--predictions_output', type=str, default='predictions.csv', help='Path to the file in which to save the results.')
parser.add_argument('--test_appearances_ids', nargs="+", type=int, default=[76, 118, 784, 418, 995, 478, 425, 273, 967, 610],
                    help="Provide a list of integers separated by spaces.")


# each LightningModule defines arguments relevant to it
args = parser.parse_args()

if not args.openpose and not args.holistic:
    raise Exception("Load at least one pose system")

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

pose_headers = []

HOLISTIC_POSE_HEADER = OPENPOSE_POSE_HEADER = None

if args.holistic:
    with open("holistic.poseheader", "rb") as f:
        HOLISTIC_POSE_HEADER = PoseHeader.read(BufferReader(f.read()))

    components = []
    for c in HOLISTIC_POSE_HEADER.components:
        if c.name in args.holistic_pose_components:
            c_copy = pickle.loads(pickle.dumps(c))
            c_copy.format = c_copy.format[:args.pose_dims]
            components.append(c_copy)

    rep_header = PoseHeader(version=HOLISTIC_POSE_HEADER.version, dimensions=HOLISTIC_POSE_HEADER.dimensions,
                            components=components)
    pose_headers.append(rep_header)

if args.openpose:
    with open("openpose_135.poseheader", "rb") as f:
        OPENPOSE_POSE_HEADER = PoseHeader.read(BufferReader(f.read()))
        OPENPOSE_POSE_HEADER.components[0].format = "XY"
    pose_headers.append(OPENPOSE_POSE_HEADER)

if len(pose_headers) == 1:
    POSE_HEADER = pose_headers[0]
else:
    POSE_HEADER = PoseHeader(version=pose_headers[0].version, dimensions=pose_headers[0].dimensions,
                             components=pose_headers[0].components + pose_headers[1].components)

POSE_REP = TorchPoseRepresentation(header=POSE_HEADER, rep_modules1=rep_modules1, rep_modules2=rep_modules2)

COMPONENTS = []
FLIPPED_COMPONENTS = []

if args.holistic:
    COMPONENTS = list(args.holistic_pose_components)

    FLIPPED_COMPONENTS = list(args.holistic_pose_components)
    FLIPPED_COMPONENTS[COMPONENTS.index("LEFT_HAND_LANDMARKS")] = "RIGHT_HAND_LANDMARKS"
    FLIPPED_COMPONENTS[COMPONENTS.index("RIGHT_HAND_LANDMARKS")] = "LEFT_HAND_LANDMARKS"

if args.openpose:
    COMPONENTS.append("BODY_135")
    FLIPPED_COMPONENTS.append("BODY_135")

# ---------------------
# Set Seed
# ---------------------
if args.seed == 0:  # Make seed random if 0
    args.seed = random.randint(0, 1000)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
