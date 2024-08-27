import random
from typing import Dict, List

import numpy as np
import numpy.ma as ma
# noinspection PyUnresolvedReferences
import sign_language_datasets.datasets
import tensorflow_datasets as tfds
import torch
from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.torch.masked import MaskedTensor, MaskedTorch
from pose_format.utils.holistic import FLIPPED_BODY_POINTS
from sign_language_datasets.datasets.config import SignDatasetConfig
from torch.utils.data import Dataset
from tqdm import tqdm

from pose_anonymization.appearance import remove_appearance, transfer_appearance

from args import args, FLIPPED_COMPONENTS, HOLISTIC_POSE_HEADER, OPENPOSE_POSE_HEADER, POSE_HEADER


class ZeroPadCollator:
    @staticmethod
    def collate_tensors(batch: List[torch.Tensor]) -> torch.Tensor:
        if isinstance(batch[0], int) or isinstance(batch[0], np.int32):
            return torch.tensor(batch, dtype=torch.long)

        if isinstance(batch[0], str) or isinstance(batch[0], bytes):
            return batch

        max_len = max([len(t) for t in batch])
        if max_len == 1:
            return torch.stack(batch)

        torch_cls = MaskedTorch if isinstance(batch[0], MaskedTensor) else torch

        new_batch = []
        for t in batch:
            missing = list(t.shape)
            missing[0] = max_len - t.shape[0]

            if missing[0] > 0:
                padding_tensor = torch.zeros(missing, dtype=t.dtype, device=t.device)
                t = torch_cls.cat([t, padding_tensor], dim=0)

            new_batch.append(t)

        return torch_cls.stack(new_batch, dim=0)

    def collate(self, batch, ) -> Dict[str, torch.Tensor]:
        keys = batch[0].keys()
        return {k: self.collate_tensors([b[k] for b in batch]) for k in keys}


class PoseClassificationMockDataset(Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, index):
        x = torch.randn((random.randint(10, 20), 1, POSE_HEADER.total_points(), args.pose_dims), dtype=torch.float)

        return {
            "id": "some_id",
            "signer": 0,
            "pose": MaskedTensor(x),
            "label": 0
        }


class PoseClassificationDataset(Dataset):
    def __init__(self, data: List, is_train=False, anonymize=False, transfer_appearance=False, signers_poses={}):
        self.data = data
        self.is_train = is_train
        self.anonymize = anonymize
        self.transfer_appearance = transfer_appearance
        self.signers_poses = signers_poses

    @staticmethod
    def from_tfds(tf_dataset, pose_type: str, anonymize: bool = False, transfer_appearance: bool = False, **kwargs):
        data = []

        header = HOLISTIC_POSE_HEADER if pose_type == "holistic" else OPENPOSE_POSE_HEADER

        normalization_info = header.normalization_info(
            p1=("POSE_LANDMARKS", "RIGHT_SHOULDER"),
            p2=("POSE_LANDMARKS", "LEFT_SHOULDER")
        ) if pose_type == "holistic" else header.normalization_info(
            p1=("BODY_135", "RShoulder"),
            p2=("BODY_135", "LShoulder")
        )

        signers_poses = {}

        for datum in tqdm(tf_dataset):
            body = NumPyPoseBody(30, datum["pose"]["data"].numpy(), datum["pose"]["conf"].numpy())

            pose = Pose(header, body)
            # Only use requested dimensions
            if pose.body.data.shape[-1] != args.pose_dims:
                pose.body.data = pose.body.data[:, :, :, :args.pose_dims]

            # Get subset of components if needed
            if pose_type == "holistic" and len(args.holistic_pose_components) < 4:
                pose = pose.get_components(args.holistic_pose_components)

            pose = pose.normalize(normalization_info)
            # new_fps = (args.seq_size / len(pose.body.data)) * 30
            # pose = pose.interpolate(new_fps=new_fps, kind='linear')

            gloss_id = datum["gloss_id"].numpy()
            data.append({
                "id": datum["id"].numpy(),
                "signer": datum["signer"].numpy(),
                "pose": pose,
                "label": gloss_id if gloss_id > 0 else 0
            })
            if datum["signer"].numpy() not in signers_poses:
                signers_poses[datum["signer"].numpy()] = pose

        kwargs['anonymize'] = anonymize
        kwargs['transfer_appearance'] = transfer_appearance
        kwargs['signers_poses'] = signers_poses
        return PoseClassificationDataset(data, **kwargs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        datum = self.data[index]
        pose = datum["pose"]

        if self.anonymize:
            pose = remove_appearance(pose)

        elif self.transfer_appearance:
            key = random.choice(list(self.signers_poses.keys()))
            target_appearance_pose = self.signers_poses[key]
            #target_appearance_pose = self.signers_poses[random.randint(0, len(self.signers_poses) - 1)]
            pose = transfer_appearance(pose=pose, appearance_pose=target_appearance_pose)
        torch_body_data = pose.body.torch().data

        return {
            "id": datum["id"],
            "signer": datum["signer"],
            "pose": torch_body_data,
            "length": torch.ones((len(torch_body_data)), dtype=torch.bool),
            "label": datum["label"]
        }


def get_autsl(split: str, anonymize: bool = False, transfer_appearance: bool = False):
    datasets = []
    if args.holistic:
        datasets.append(get_autsl_format(split=split, pose="holistic", anonymize=anonymize, transfer_appearance=transfer_appearance))
    if args.openpose:
        datasets.append(get_autsl_format(split=split, pose="openpose", anonymize=anonymize, transfer_appearance=transfer_appearance))

    if len(datasets) > 1:
        for dataset in datasets:
            dataset.data = sorted(dataset.data, key=lambda d: d["id"].decode('utf-8'))

        for datum1, datum2 in tqdm(zip(datasets[0].data, datasets[1].data)):
            assert datum1["id"].decode('utf-8') == datum2["id"].decode('utf-8')
            data = ma.concatenate([datum1["pose"].body.data, datum2["pose"].body.data], axis=2)
            conf = ma.concatenate([datum1["pose"].body.confidence, datum2["pose"].body.confidence], axis=2)
            pose_body = NumPyPoseBody(fps=datum1["pose"].body.fps, data=data, confidence=conf)
            datum1["pose"] = Pose(header=POSE_HEADER, body=pose_body)

    return datasets[0]


def get_autsl_format(split: str, pose: str, anonymize: bool = False, transfer_appearance: bool = False):
    name = "poses-new-" + str(args.fps) if pose == "holistic" else "openpose-" + str(args.fps)
    config = SignDatasetConfig(name=name, include_video=False, include_pose=pose, fps=args.fps)
    data_set = tfds.load(
        'autsl',
        builder_kwargs=dict(config=config), # train_decryption_key="", valid_decryption_key="", test_decryption_key=""
        split=split,
        shuffle_files=False,
        as_supervised=False,
    )

    return PoseClassificationDataset.from_tfds(data_set, pose, anonymize, transfer_appearance)

# WARNING:root:signers_poses: dict_keys([21, 8, 3, 38, 4, 10, 12, 7, 26, 13, 17, 42, 41, 33, 2, 24, 0, 5, 32, 15, 22, 29, 36, 23, 9, 19, 40, 28, 31, 37, 20])
def split_train_dataset(dataset: PoseClassificationDataset, ids):
    ids = set(ids)
    train_data = [d for d in dataset.data if d["signer"] not in ids]
    valid_data = [d for d in dataset.data if d["signer"] in ids]
    anonymize = dataset.anonymize
    transfer_appearance = dataset.transfer_appearance
    signers_poses = dataset.signers_poses

    return PoseClassificationDataset(train_data, is_train=True, anonymize=anonymize, transfer_appearance=transfer_appearance, signers_poses=signers_poses), PoseClassificationDataset(valid_data, is_train=False, anonymize=anonymize, transfer_appearance=transfer_appearance, signers_poses=signers_poses)


if __name__ == "__main__":
    split_type = "train"  # or 'test' or 'validation' depending on your needs
    anonymize=False
    transfer_appearance=False
    get_autsl(split=split_type, anonymize=anonymize, transfer_appearance=transfer_appearance)
