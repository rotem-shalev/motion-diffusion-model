# import pickle as pkl
import json
import numpy as np
from math import inf
import random
from collections import defaultdict
import os
import sys
from torch.utils.data import Dataset
import torch
from utils.misc import to_torch
import utils.rotation_conversions as geometry

rootdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, rootdir)


class InterHand(Dataset):
    dataname = "interhand"

    def __init__(self, datapath="dataset/interHand", split="train", fps=30, only_right=True, translation=True,
                 align_pose_frontview=False, num_frames=1, min_len=-1, max_len=-1, max_seq_num=inf, sampling="conseq",
                 sampling_step=1, glob=True, pose_rep="rot6d"):

        super().__init__()

        if split not in ["train", "val", "test"]:
            raise ValueError(f"{split} is not a valid split")

        self.split = split

        # with open(f"/mnt/raid1/home/rotem_shalev/annotations/annotations/{self.split}/InterHand2.6M_"
        #           f"{self.split}_joint_3d.json",
        #           'r') as f1:
        #     data_3d = json.load(f1)

        self.num_frames = num_frames
        self.datapath = os.path.join(rootdir, datapath)
        # TODO- read all splits and split again by cap id and not by frames! (check what does sequence name mean-
        #  maybe its a split inside cap?)

        seq_frames_ids = defaultdict(lambda: defaultdict(set))
        with open(f"/mnt/raid1/home/rotem_shalev/annotations/annotations/{self.split}/InterHand2.6M_"
                  f"{self.split}_data.json", 'r') as f:
            json_data = json.load(f)
            for img in json_data['images']:
                seq_frames_ids[str(img['capture'])][img['seq_name']].add(str(img['frame_idx']))

        datafilepath = os.path.join(self.datapath, f"{fps}fps_annotations/"
                                    f"{self.split}/InterHand2.6M_{self.split}_MANO_NeuralAnnot.json")

        with open(datafilepath, 'r') as f:
            data = json.load(f)

        seq_annots = defaultdict(list)
        for cap, seq in seq_frames_ids.items():
            for seq_name, frames in seq.items():
                if len(frames) < 40:
                    continue
                frames = sorted(frames)
                cur_seq = []
                for frame_idx in frames:
                    if cap in data and frame_idx in data[cap] and data[cap][frame_idx]['right'] is not None:
                        cur_seq.append(data[cap][frame_idx]['right'])
                        if cap not in data_3d or frame_idx not in data_3d[cap]:
                            print(f"not in data_3d: cap: {cap}, frame_idx: {frame_idx}")
                    else:
                        if cap not in data or frame_idx not in data[cap]:
                            print(f"not in data: cap: {cap}, frame_idx: {frame_idx}")
                            if cap in data_3d and frame_idx in data_3d[cap]:
                                print("in data_3d")
                        if cur_seq:
                            if len(cur_seq) >= 40:
                                seq_annots[seq_name].append(cur_seq)
                            cur_seq = []
                if len(cur_seq) >= 40:
                    seq_annots[seq_name].append(cur_seq)

        self._pose = []
        self._trans = []
        for seq_list in seq_annots.values():
            for seq in seq_list:
                for i in range(0, len(seq)//self.num_frames, self.num_frames):
                    cur_pose = []
                    cur_trans = []
                    for j in range(self.num_frames):
                        cur_pose.append(seq[i+j]['pose'])
                        cur_trans.append(seq[i+j]['trans'])
                    self._pose.append(np.array(cur_pose))
                    self._trans.append(np.array(cur_trans))

        # data structure:
        # |-- str (capture id)
        # |   |-- str (frame idx):
        # |   |   |-- 'right'
        # |   |   |   |-- 'pose': 48 dimensional MANO pose vector in axis-angle representation minus the mean pose.
        # |   |   |   |-- 'shape': 10 dimensional MANO shape vector.
        # |   |   |   |-- 'trans': 3 dimensional MANO translation vector in meter unit.
        # |   |   |-- 'left'
        # |   |   |   |-- 'pose': 48 dimensional MANO pose vector in axis-angle representation minus the mean pose.
        # |   |   |   |-- 'shape': 10 dimensional MANO shape vector.
        # |   |   |   |-- 'trans': 3 dimensional MANO translation vector in meter unit.

        # for cap_id, frames in data.items():
        #     cur_pose = []
        #     cur_shape = []
        #     cur_trans = []
        #     cur_num_frames = 0
        #     # sorted_frames = dict(sorted(frames.items(), key=lambda x: int(x[0])))
        #     for frame_idx, hands in frames.items():
        #         if only_right:
        #             if hands['right'] is None:
        #                 break
        #             cur_pose.append(hands['right']['pose'])
        #             cur_shape.append(hands['right']['shape'])
        #             cur_trans.append(hands['right']['trans'])
        #         else:
        #             if hands['left'] is not None:
        #                 cur_pose.append((hands['right']['pose'], hands['left']['pose']))
        #                 cur_shape.append((hands['right']['shape'], hands['left']['shape']))
        #                 cur_trans.append((hands['right']['trans'], hands['left']['trans']))
        #             else:
        #                 cur_pose.append((hands['right']['pose'], None))
        #                 cur_shape.append((hands['right']['shape'], None))
        #                 cur_trans.append((hands['right']['trans'], None))
        #         cur_num_frames += 1
        #         if cur_num_frames == self.num_frames:
        #             self._pose.append(np.array(cur_pose))
        #             self._shape.append(np.array(cur_shape))
        #             self._trans.append(np.array(cur_trans))
        #             # self._num_frames_in_video.append(cur_num_frames)
        #             cur_num_frames = 0
        #             cur_pose = []
        #             cur_shape = []
        #             cur_trans = []
        #     if cur_num_frames == num_frames:
        #         self._pose.append(np.array(cur_pose))
        #         self._shape.append(np.array(cur_shape))
        #         self._trans.append(np.array(cur_trans))
                # self._num_frames_in_video.append(cur_num_frames)

        self._data_ind = list(range(len(self._pose)))
        self.split = split
        self.max_seq_num = max_seq_num  # for testing
        self.min_len = min_len
        self.max_len = max_len
        self.glob = glob
        self.align_pose_frontview = align_pose_frontview
        self.translation = translation
        self.sampling = sampling
        self.sampling_step = sampling_step
        self.pose_rep = pose_rep

    def __getitem__(self, index):
        data_index = self._data_ind[index]
        return self._get_item_data_index(data_index)

    def __len__(self):
        max_seq_num = getattr(self, "max_seq_num", inf)
        return min(len(self._data_ind), max_seq_num)

    # def _load_joints3D(self, ind, frame_ix):
    #     return self._joints[ind][frame_ix]

    def _load_rotvec(self, ind, frame_ix):
        pose = self._pose[ind][frame_ix].reshape(-1, 16, 3)  # MANO structure, 15 joints + 1 global orientation
        return pose

    def get_pose_data(self, data_index, frame_ix):
        pose = self._load(data_index, frame_ix)
        return pose

    def _load(self, ind, frame_ix):
        if not hasattr(self, "_load_rotvec"):
            raise ValueError("load rotation vector is not implemented.")
        else:
            pose = self._load_rotvec(ind, frame_ix)
            if not self.glob:
                pose = pose[:, 1:, :]
            pose = to_torch(pose)
            ret = pose

        ret_tr = to_torch(self._trans[ind][frame_ix])

        if self.pose_rep == "rot6d":
            ret = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(pose))

        if self.translation:  # pad global translation to 6d and concat it to ret
            padded_tr = torch.zeros((ret.shape[0], ret.shape[2]), dtype=ret.dtype)
            padded_tr[:, :3] = ret_tr
            ret = torch.cat((ret, padded_tr[:, None]), 1)

        ret = ret.permute(1, 2, 0).contiguous()
        return ret.float()

    def _get_item_data_index(self, data_index):
        # nframes = self._num_frames_in_video[data_index]

        # if self.num_frames == -1 and self.max_len == -1:
        frame_ix = np.arange(self.num_frames)
        # else:
        #     if self.num_frames == -2:
        #         if self.min_len <= 0:
        #             raise ValueError("You should put a min_len > 0 for num_frames == -2 mode")  # TODO what is -2 mode?
        #         if self.max_len != -1:
        #             max_frame = min(self.num_frames, self.max_len)
        #         else:
        #             max_frame = self.num_frames
        #
        #         num_frames = random.randint(self.min_len, max(max_frame, self.min_len))
        #
        #     else:
        #         num_frames = self.num_frames if self.num_frames != -1 else self.max_len
        #
        #     if num_frames > self.num_frames:
        #         fair = False  # True
        #         if fair:
        #             # distills redundancy everywhere
        #             choices = np.random.choice(range(self.num_frames),
        #                                        num_frames,
        #                                        replace=True)
        #             frame_ix = sorted(choices)
        #         else:
        #             # adding the last frame until done
        #             ntoadd = max(0, num_frames - nframes)
        #             lastframe = nframes - 1
        #             padding = lastframe * np.ones(ntoadd, dtype=int)
        #             frame_ix = np.concatenate((np.arange(0, nframes),
        #                                        padding))

            # elif self.sampling in ["conseq", "random_conseq"]:
            #     step_max = (nframes - 1) // (num_frames - 1)
            #     if self.sampling == "conseq":
            #         if self.sampling_step == -1 or self.sampling_step * (num_frames - 1) >= nframes:
            #             step = step_max
            #         else:
            #             step = self.sampling_step
            #     elif self.sampling == "random_conseq":
            #         step = random.randint(1, step_max)

            #     lastone = step * (num_frames - 1)
            #     shift_max = nframes - lastone - 1
            #     shift = random.randint(0, max(0, shift_max - 1))
            #     frame_ix = shift + np.arange(0, lastone + 1, step)
            #
            # elif self.sampling == "random":
            #     choices = np.random.choice(range(nframes),
            #                                num_frames,
            #                                replace=False)
            #     frame_ix = sorted(choices)

            # else:
            #     raise ValueError("Sampling not recognized.")

        inp = self.get_pose_data(data_index, frame_ix)
        output = {'inp': inp}
        return output



