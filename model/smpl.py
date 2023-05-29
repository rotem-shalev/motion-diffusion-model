# This code is based on https://github.com/Mathux/ACTOR.git
import numpy as np
import torch

import contextlib

from smplx import SMPLLayer as _SMPLLayer
from smplx import MANOLayer as _MANOLayer
from smplx.lbs import vertices2joints

# change 0 and 8
action2motion_joints = [8, 1, 2, 3, 4, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 21, 24, 38]

from utils.config import SMPL_MODEL_PATH, JOINT_REGRESSOR_TRAIN_EXTRA, SMPL_MANO_MODEL_PATH

JOINTSTYPE_ROOT = {"a2m": 0,  # action2motion
                   "smpl": 0,
                   "a2mpl": 0,  # set(smpl, a2m)
                   "vibe": 8}  # 0 is the 8 position: OP MidHip below

HANDS_MAP = {
    'r_thumb4': 0, 'r_thumb3': 1, 'r_thumb2': 2, 'r_thumb1': 3, 'r_index4': 4, 'r_index3': 5, 'r_index2': 6,
    'r_index1': 7, 'r_middle4': 8, 'r_middle3': 9, 'r_middle2': 10, 'r_middle1': 11, 'r_ring4': 12,
    'r_ring3': 13, 'r_ring2': 14, 'r_ring1': 15, 'r_pinky4': 16, 'r_pinky3': 17, 'r_pinky2': 18,
    'r_pinky1': 19, 'r_wrist': 20, 'l_thumb4': 21, 'l_thumb3': 22, 'l_thumb2': 23, 'l_thumb1': 24,
    'l_index4': 25, 'l_index3': 26, 'l_index2': 27, 'l_index1': 28, 'l_middle4': 29, 'l_middle3': 30,
    'l_middle2': 31, 'l_middle1': 32, 'l_ring4': 33, 'l_ring3': 34, 'l_ring2': 35, 'l_ring1': 36,
    'l_pinky4': 37, 'l_pinky3': 38, 'l_pinky2': 39, 'l_pinky1': 40, 'l_wrist': 41
}

HANDS_NAMES = [
    'r_thumb4', 'r_thumb3', 'r_thumb2', 'r_thumb1', 'r_index4', 'r_index3', 'r_index2',
    'r_index1', 'r_middle4', 'r_middle3', 'r_middle2', 'r_middle1', 'r_ring4',
    'r_ring3', 'r_ring2', 'r_ring1', 'r_pinky4', 'r_pinky3', 'r_pinky2',
    'r_pinky1', 'r_wrist', 'l_thumb4', 'l_thumb3', 'l_thumb2', 'l_thumb1',
    'l_index4', 'l_index3', 'l_index2', 'l_index1', 'l_middle4', 'l_middle3',
    'l_middle2', 'l_middle1', 'l_ring4', 'l_ring3', 'l_ring2', 'l_ring1',
    'l_pinky4', 'l_pinky3', 'l_pinky2', 'l_pinky1', 'l_wrist'
]


R_HAND_MAP = {
    'r_wrist': 0,
    'r_thumb1': 1, 'r_thumb2': 2, 'r_thumb3': 3,
    'r_index1': 4, 'r_index2': 5, 'r_index3': 6,
    'r_middle1': 7, 'r_middle2': 8, 'r_middle3': 9,
    'r_ring1': 10, 'r_ring2': 11, 'r_ring3': 12,
    'r_pinky1': 13, 'r_pinky2': 14, 'r_pinky3': 15
}

R_HAND_NAMES = [
    'r_wrist',
    'r_thumb1', 'r_thumb2', 'r_thumb3',
    'r_index1', 'r_index2', 'r_index3',
    'r_middle1', 'r_middle2', 'r_middle3',
    'r_ring1', 'r_ring2', 'r_ring3',
    'r_pinky1', 'r_pinky2', 'r_pinky3'
]


JOINT_MAP = {
    'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17,
    'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16,
    'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0,
    'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8,
    'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7,
    'OP REye': 25, 'OP LEye': 26, 'OP REar': 27,
    'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
    'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
    'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45,
    'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7,
    'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17,
    'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20,
    'Neck (LSP)': 47, 'Top of Head (LSP)': 48,
    'Pelvis (MPII)': 49, 'Thorax (MPII)': 50,
    'Spine (H36M)': 51, 'Jaw (H36M)': 52,
    'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26,
    'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27
}

JOINT_NAMES = [
    'OP Nose', 'OP Neck', 'OP RShoulder',
    'OP RElbow', 'OP RWrist', 'OP LShoulder',
    'OP LElbow', 'OP LWrist', 'OP MidHip',
    'OP RHip', 'OP RKnee', 'OP RAnkle',
    'OP LHip', 'OP LKnee', 'OP LAnkle',
    'OP REye', 'OP LEye', 'OP REar',
    'OP LEar', 'OP LBigToe', 'OP LSmallToe',
    'OP LHeel', 'OP RBigToe', 'OP RSmallToe', 'OP RHeel',
    'Right Ankle', 'Right Knee', 'Right Hip',
    'Left Hip', 'Left Knee', 'Left Ankle',
    'Right Wrist', 'Right Elbow', 'Right Shoulder',
    'Left Shoulder', 'Left Elbow', 'Left Wrist',
    'Neck (LSP)', 'Top of Head (LSP)',
    'Pelvis (MPII)', 'Thorax (MPII)',
    'Spine (H36M)', 'Jaw (H36M)',
    'Head (H36M)', 'Nose', 'Left Eye',
    'Right Eye', 'Left Ear', 'Right Ear'
]


# adapted from VIBE/SPIN to output smpl_joints, vibe joints and action2motion joints
# class SMPL(_SMPLLayer):
class SMPL(_MANOLayer):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, model_path=SMPL_MODEL_PATH, **kwargs):
        self.hands = "MANO" in model_path
        self.model_path = model_path
        kwargs["model_path"] = model_path

        # remove the verbosity for the 10-shapes beta parameters
        with contextlib.redirect_stdout(None):
            super(SMPL, self).__init__(**kwargs)

        if self.hands:
            vibe_indexes = np.array([R_HAND_MAP[i] for i in R_HAND_NAMES])
            smpl_indexes = np.arange(16)
            self.maps = {"vibe": vibe_indexes,
                         "smpl": smpl_indexes
                         }
        else:
            J_regressor_extra = np.load(JOINT_REGRESSOR_TRAIN_EXTRA)
            self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))
            vibe_indexes = np.array([JOINT_MAP[i] for i in JOINT_NAMES])
            a2m_indexes = vibe_indexes[action2motion_joints]
            smpl_indexes = np.arange(24)
            a2mpl_indexes = np.unique(np.r_[smpl_indexes, a2m_indexes])

            self.maps = {"vibe": vibe_indexes,
                         "a2m": a2m_indexes,
                         "smpl": smpl_indexes,
                         "a2mpl": a2mpl_indexes
                         }

    def forward(self, *args, **kwargs):
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        if self.hands:
            all_joints = smpl_output.joints
        else:
            extra_joints = vertices2joints(self.J_regressor, smpl_output.vertices)
            all_joints = torch.cat([smpl_output.joints, extra_joints], dim=1)

        output = {"vertices": smpl_output.vertices}

        for joinstype, indexes in self.maps.items():
            output[joinstype] = all_joints[:, indexes]

        return output
