import pickle as pkl
from math import inf
import os
import sys
from torch.utils.data import Dataset
import torch
from data_loaders.ham2pose.hamnosys_tokenizer import HamNoSysTokenizer

rootdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, rootdir)

MIN_CONFIDENCE = 0.2


class Ham2Pose(Dataset):
    dataname = "ham2pose"

    def __init__(self, split="train", num_frames=inf, min_len=1, max_len=200, max_seq_num=inf,
                 sampling="conseq", sampling_step=1, pose_rep="xyz", use_how2sign=False, augment_rate=0.0,
                 conf_power=1.0):

        super().__init__()

        if split not in ["train", "val", "test"]:
            raise ValueError(f"{split} is not a valid split")

        self.split = split
        self.max_len = max_len
        self.max_seq_num = max_seq_num  # allow limitation for testing

        with open("/mnt/raid1/home/rotem_shalev/motion-diffusion-model/dataset/ham2pose_processed_dataset_3.pkl",
                  'rb') as f:
            data = pkl.load(f)
        self.data = data[split]

        if use_how2sign:
            with open("/mnt/raid1/home/rotem_shalev/motion-diffusion-model/dataset/how2sign_processed_dataset_2.pkl",
                      'rb') as f:
                how2sign_data = pkl.load(f)
            self.data += how2sign_data

        if max_seq_num < len(self.data):
            self._data_ind = list(range(max_seq_num))
        else:
            self._data_ind = list(range(len(self.data)))

        self.min_len = min_len
        self.max_len = max_len
        self.sampling = sampling
        self.sampling_step = sampling_step
        self.augment_rate = augment_rate
        self.pose_rep = pose_rep
        self.conf_power = conf_power

    def __getitem__(self, index):
        data_index = self._data_ind[index]
        return self._get_item_data_index(data_index)

    def __len__(self):
        max_seq_num = getattr(self, "max_seq_num", inf)
        return min(len(self._data_ind), max_seq_num)

    def get_pose_data(self, data_index):
        pose, conf = self._load(data_index)
        return pose, conf

    def _load(self, ind):
        pose_data = self.data[ind]['pose'].body
        pose = torch.from_numpy(pose_data.data).squeeze(1)
        conf = torch.from_numpy(pose_data.confidence).squeeze(1)
        conf = conf**self.conf_power
        pose = pose.permute(1, 2, 0).contiguous()
        conf = conf.permute(1, 0).contiguous()
        return pose, conf

    def _get_item_data_index(self, data_index):
        pose, conf = self.get_pose_data(data_index)
        text = self.data[data_index]["hamnosys"]
        output = {'inp': pose, 'confidence': conf, "text": text, "id": self.data[data_index]["id"]}
        return output

