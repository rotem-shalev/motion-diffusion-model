import os
import sys
import numpy as np
import glob
import json
import pickle as pkl
from collections import defaultdict
from data_loaders.grab.utils import parse_npz

rootdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, rootdir)


def preprocess_hanco(data_path="/mnt/raid1/home/rotem_shalev/HanCO/shape", min_len=1):
    _pose = []
    _trans = []
    data_dirs = os.listdir(data_path)
    for dir in data_dirs:
        frames = sorted(os.listdir(os.path.join(data_path, dir)))
        if len(frames) < min_len + 8:  # min_len frames + 8 cam folders
            continue
        cur_pose = []
        cur_trans = []
        for frame_json_path in frames:
            json_path = os.path.join(data_path, dir, frame_json_path)
            if not os.path.isfile(json_path):
                continue
            with open(json_path, 'r') as f:
                frame_json = json.load(f)
                cur_pose.append(np.array(frame_json["poses"][0]))
                cur_trans.append(np.array(frame_json["global_t"][0][0]))

        _pose.append((np.array(cur_pose), np.array([])))
        _trans.append((np.array(cur_trans), np.array([])))

    return {"pose": np.array(_pose), "trans": np.array(_trans)}


def get_frame_data(task_path, hand_path, prev_pose_dic=None):
    all_frames = os.listdir(task_path)
    all_frames = sorted(all_frames, key=lambda x: int(x[:-7]))
    pose_dic = dict()
    cur_pose = []
    cur_trans = []

    for frame_path in all_frames:
        full_frame_path = os.path.join(task_path, frame_path)
        with open(full_frame_path, 'rb') as frame:
            frame_data = pkl.load(frame, encoding='latin1')

        if not isinstance(frame_data["poseCoeff"], np.ndarray):
            # if len(cur_pose) > 0: # only 12 in dataset
                # _pose.append(np.array(cur_pose))
                # _trans.append(np.array(cur_trans))
            cur_pose = []
            cur_trans = []
            pose_dic = dict()
        else:
            pose_dic[os.path.join(hand_path, frame_path)] = len(cur_pose)
            cur_pose.append(frame_data["poseCoeff"])
            cur_trans.append(frame_data["trans"])
    if len(cur_pose) > 0 and prev_pose_dic and len(prev_pose_dic) != len(pose_dic):
        ret_pose = []
        ret_trans = []
        for key in prev_pose_dic:
            if key in pose_dic:
                ret_pose.append(cur_pose[pose_dic[key]])
                ret_trans.append(cur_trans[pose_dic[key]])
            else:
                ret_pose.append(np.array([]))
                ret_trans.append(np.array([]))
        return None, ret_pose, ret_trans

    return pose_dic, cur_pose, cur_trans


def preprocess_hoi4d(data_path="/mnt/raid1/home/rotem_shalev/HOI4D/handpose", min_len=1):
    _pose = []
    _trans = []

    humans = os.listdir(data_path + f"/refinehandpose_right")
    for human in humans:
        human_path = os.path.join(data_path, f"refinehandpose_right", human)
        human_path_left = os.path.join(data_path, f"refinehandpose_left", human)
        all_object_classes = os.listdir(human_path)
        for obj in all_object_classes:
            all_object_ids = os.listdir(os.path.join(human_path, obj))
            for obj_id in all_object_ids:
                all_rooms = os.listdir(os.path.join(human_path, obj, obj_id))
                for room_id in all_rooms:
                    all_room_layouts = os.listdir(os.path.join(human_path, obj, obj_id, room_id))
                    for room_layout in all_room_layouts:
                        all_tasks = os.listdir(os.path.join(human_path, obj, obj_id, room_id, room_layout))
                        for task in all_tasks:
                            task_path = os.path.join(human_path, obj, obj_id, room_id, room_layout, task)
                            hand_path = os.path.join(human, obj, obj_id, room_id, room_layout, task)
                            cur_pose_dic, cur_pose, cur_trans = get_frame_data(task_path, hand_path)
                            task_path_left = os.path.join(human_path_left, obj, obj_id, room_id, room_layout, task)
                            if cur_pose_dic and os.path.isdir(task_path_left):
                                _, cur_lpose, cur_ltrans = get_frame_data(task_path_left, hand_path, cur_pose_dic)
                            else:
                                cur_lpose, cur_ltrans = [], []

                            if len(cur_pose) >= min_len:
                                _pose.append((np.array(cur_pose), np.array(cur_lpose)))
                                _trans.append((np.array(cur_trans), np.array(cur_ltrans)))

    return {"pose": np.array(_pose), "trans": np.array(_trans)}


def change_fps(sampling_step, pose, trans):
    pose = pose[::sampling_step, :]
    trans = trans[::sampling_step, :]
    return pose, trans


def preprocess_grab(data_path="/mnt/raid1/home/rotem_shalev/GRAB_dataset/GRAB_unzipped/grab", sampling_step=4):
    _pose = []
    _trans = []
    _global_orient = []

    all_seqs = glob.glob(data_path + '/*/*.npz')
    for sequence in all_seqs:
        seq_data = parse_npz(sequence)
        cur_rpose = seq_data.rhand.params.fullpose  # (n_frames, 45)
        cur_rtrans = seq_data.rhand.params.transl  # (n_frames, 3)
        cur_rglobal_orient = seq_data.rhand.params.global_orient  # (n_frames, 3)
        cur_rpose = np.concatenate((cur_rpose, cur_rglobal_orient), axis=1)
        # reduce FPS (originally its 120)
        cur_rpose, cur_rtrans = change_fps(sampling_step, cur_rpose, cur_rtrans)

        cur_lpose = seq_data.lhand.params.fullpose
        cur_ltrans = seq_data.lhand.params.transl
        cur_lglobal_orient = seq_data.lhand.params.global_orient
        cur_lpose = np.concatenate((cur_lpose, cur_lglobal_orient), axis=1)
        cur_lpose, cur_ltrans = change_fps(sampling_step, cur_lpose, cur_ltrans)

        _pose.append((cur_rpose, cur_lpose))
        _trans.append((cur_rtrans, cur_ltrans))
    return {"pose": np.array(_pose), "trans": np.array(_trans)}


def preprocess_interhand(data_path="dataset/interHand", min_len=1):
    _pose = []
    _trans = []
    data_path = os.path.join(rootdir, data_path)

    seq_frames_ids = defaultdict(lambda: defaultdict(set))
    for split in ["train", "val", "test"]:
        with open(f"/mnt/raid1/home/rotem_shalev/annotations/annotations/{split}/InterHand2.6M_{split}_data.json",
                  'r') as f:
            json_data = json.load(f)
            for img in json_data['images']:
                seq_frames_ids[str(img['capture'])][img['seq_name']].add(str(img['frame_idx']))

    datafilepath = os.path.join(data_path, f"30fps_annotations/{split}/InterHand2.6M_{split}_MANO_NeuralAnnot.json")

    with open(datafilepath, 'r') as f:
        data = json.load(f)

    seq_annots = defaultdict(lambda: defaultdict(list))

    for cap, seq in seq_frames_ids.items():
        for seq_name, frames in seq.items():
            if len(frames) < min_len:
                continue
            frames = sorted(frames)
            cur_rseq_pose = []
            cur_rseq_trans = []
            cur_lseq_pose = []
            cur_lseq_trans = []
            for frame_idx in frames:
                if cap in data and frame_idx in data[cap]:
                    if data[cap][frame_idx]['right'] is not None:
                        cur_rseq_pose.append(data[cap][frame_idx]['right']['pose'])
                        cur_rseq_trans.append(data[cap][frame_idx]['right']['trans'])
                    if data[cap][frame_idx]['left'] is not None:
                        cur_lseq_pose.append(np.array(data[cap][frame_idx]['left']['pose']))
                        cur_lseq_trans.append(np.array(data[cap][frame_idx]['left']['trans']))
                elif cur_rseq_pose or cur_lseq_pose:
                    if len(cur_rseq_pose) >= min_len or len(cur_lseq_pose) >= min_len:
                        seq_annots[seq_name]['pose'].append((np.array(cur_rseq_pose), np.array(cur_lseq_pose)))
                        seq_annots[seq_name]['trans'].append((np.array(cur_rseq_trans), np.array(cur_lseq_trans)))
                    cur_rseq_pose = []
                    cur_rseq_trans = []
                    cur_lseq_pose = []
                    cur_lseq_trans = []
            if len(cur_rseq_pose) >= min_len or len(cur_lseq_pose) >= min_len:
                seq_annots[seq_name]['pose'].append((np.array(cur_rseq_pose), np.array(cur_lseq_pose)))
                seq_annots[seq_name]['trans'].append((np.array(cur_rseq_trans), np.array(cur_lseq_trans)))

    _pose = []
    _trans = []
    for seq in seq_annots.values():
        _pose += seq['pose']
        _trans += seq['trans']

    return {"pose": np.array(_pose), "trans": np.array(_trans)}


def preprocess_data(datasets):
    hoi4d = np.array([])
    hanco = np.array([])
    grab = np.array([])
    interhand = np.array([])

    if datasets == 'all' or 'hoi4d' in datasets:
        hoi4d = preprocess_hoi4d()
    if datasets == 'all' or 'hanco' in datasets:
        hanco = preprocess_hanco()
    if datasets == 'all' or 'grab' in datasets:
        grab = preprocess_grab()
    if datasets == 'all' or 'interhand' in datasets:
        interhand = preprocess_interhand()

    np.savez("/mnt/raid1/home/rotem_shalev/motion-diffusion-model/dataset/hands_processed_dataset.npz", hoi4d=hoi4d,
             hanco=hanco, grab=grab, interhand=interhand)


if __name__ == "__main__":
    datasets = ["hanco"]
    preprocess_data(datasets)