import os
import json
import pickle as pkl
import numpy as np
from numpy import ma
from pose_format.utils.openpose import load_openpose
from pose_format.utils.reader import BufferReader
from pose_format.pose import Pose
from pose_format.pose_header import PoseHeader
from pose_format.numpy.pose_body import NumPyPoseBody
from sklearn.model_selection import train_test_split


MIN_CONFIDENCE = 0 #0.2
NUM_FACE_KEYPOINTS = 70
MAX_SEQ_LEN = 200


def swap_coords(pose, idx1, idx2):
    if type(idx1) == tuple:
        pose.body.data[:, :, idx1[0]:idx1[1]], pose.body.data[:, :, idx2[0]:idx2[1]] = \
        pose.body.data[:, :, idx2[0]:idx2[1]].copy(), pose.body.data[:, :, idx1[0]:idx1[1]].copy()
        pose.body.confidence[:, :, idx1[0]:idx1[1]], pose.body.confidence[:, :, idx2[0]:idx2[1]] = \
        pose.body.confidence[:, :, idx2[0]:idx2[1]].copy(), pose.body.confidence[:, :, idx1[0]:idx1[1]].copy()
    else:
        pose.body.data[:, :, idx1], pose.body.data[:, :, idx2] = \
            pose.body.data[:, :, idx2].copy(), pose.body.data[:, :, idx1].copy()
        pose.body.confidence[:, :, idx1], pose.body.confidence[:, :, idx2] = \
            pose.body.confidence[:, :, idx2].copy(), pose.body.confidence[:, :, idx1].copy()
    return pose


def flip_pose(pose):
    pose = pose.flip(axis=0)
    # body
    pose = swap_coords(pose, (2, 5), (5, 8))
    pose = swap_coords(pose, 9, 12)
    pose = swap_coords(pose, 15, 16)
    pose = swap_coords(pose, 17, 18)
    num_keypoints_body = 25
    # face
    for i in range(8):
        pose = swap_coords(pose, i + num_keypoints_body, 16 - i + num_keypoints_body)
    for i in range(17, 22):
        pose = swap_coords(pose, i + num_keypoints_body, 26 - i + 17 + num_keypoints_body)
    # eyes
    pose = swap_coords(pose, 36 + num_keypoints_body, 45 + num_keypoints_body)
    pose = swap_coords(pose, 41 + num_keypoints_body, 46 + num_keypoints_body)
    pose = swap_coords(pose, 40 + num_keypoints_body, 47 + num_keypoints_body)
    pose = swap_coords(pose, 37 + num_keypoints_body, 44 + num_keypoints_body)
    pose = swap_coords(pose, 38 + num_keypoints_body, 43 + num_keypoints_body)
    pose = swap_coords(pose, 39 + num_keypoints_body, 42 + num_keypoints_body)
    pose = swap_coords(pose, 68 + num_keypoints_body, 69 + num_keypoints_body)
    # nose
    pose = swap_coords(pose, 31 + num_keypoints_body, 35 + num_keypoints_body)
    pose = swap_coords(pose, 32 + num_keypoints_body, 34 + num_keypoints_body)
    # mouth
    pose = swap_coords(pose, 50 + num_keypoints_body, 52 + num_keypoints_body)
    pose = swap_coords(pose, 49 + num_keypoints_body, 53 + num_keypoints_body)
    pose = swap_coords(pose, 48 + num_keypoints_body, 54 + num_keypoints_body)
    pose = swap_coords(pose, 59 + num_keypoints_body, 55 + num_keypoints_body)
    pose = swap_coords(pose, 58 + num_keypoints_body, 56 + num_keypoints_body)
    pose = swap_coords(pose, 61 + num_keypoints_body, 63 + num_keypoints_body)
    pose = swap_coords(pose, 60 + num_keypoints_body, 64 + num_keypoints_body)
    pose = swap_coords(pose, 67 + num_keypoints_body, 65 + num_keypoints_body)
    # hands
    pose = swap_coords(pose, (-21, pose.body.data.shape[2]), (-42, -21))
    pose.body.data = pose.body.data.astype(np.float32)
    return pose


def pose_hide_low_conf(pose: Pose):
    mask = pose.body.confidence <= MIN_CONFIDENCE
    pose.body.confidence[mask] = 0
    stacked_confidence = np.stack([mask, mask], axis=3)
    masked_data = ma.masked_array(pose.body.data, mask=stacked_confidence)
    pose.body.data = masked_data


def pose_hide_legs(pose: Pose):
    if pose.header.components[0].name in ["pose_keypoints_2d", 'BODY_135']:
        point_names = ["Knee", "Ankle", "Heel", "BigToe", "SmallToe", "Hip"]
        points = [pose.header._get_point_index(pose.header.components[0].name, side+n)
                for n in point_names for side in ["L", "R"]]
    elif pose.header.components[0].name == "POSE_LANDMARKS":
        point_names = ["KNEE", "ANKLE", "HEEL", "FOOT_INDEX"]
        # pylint: disable=protected-access
        points = [pose.header._get_point_index("POSE_LANDMARKS", side + "_" + n)
                  for n in point_names for side in ["LEFT", "RIGHT"]]
    else:
        raise ValueError("Unknown pose header schema for hiding legs")

    pose.body.confidence[:, :, points] = 0
    pose.body.data[:, :, points, :] = 0


def pose_normalization_info(pose_header: PoseHeader):
    if pose_header.components[0].name == "POSE_LANDMARKS":
        return pose_header.normalization_info(
            p1=("POSE_LANDMARKS", "RIGHT_SHOULDER"),
            p2=("POSE_LANDMARKS", "LEFT_SHOULDER")
        )

    if pose_header.components[0].name == "BODY_135":
        return pose_header.normalization_info(
            p1=("BODY_135", "RShoulder"),
            p2=("BODY_135", "LShoulder")
        )

    if pose_header.components[0].name == "pose_keypoints_2d":
        return pose_header.normalization_info(
            p1=("pose_keypoints_2d", "RShoulder"),
            p2=("pose_keypoints_2d", "LShoulder")
        )

    if pose_header.components[0].name == "hand_left_keypoints_2d":
        return pose_header.normalization_info(
            p1=("hand_left_keypoints_2d", "BASE"),
            p2=("hand_right_keypoints_2d", "BASE")
        )

    raise ValueError("Unknown pose header schema for normalization")


def trim_empty_frames(pose):
    face_th = 0.5 * NUM_FACE_KEYPOINTS
    hands_th = MIN_CONFIDENCE

    # Trim all leading frames containing only zeros, almost no face or no hands
    for i in range(len(pose.body.data)):
        if pose.body.confidence[i][:, 25:-42].sum() > face_th and \
                pose.body.confidence[i][:, 4] + pose.body.confidence[i][:, 7] > hands_th:
            if i != 0:
                pose.body.data = pose.body.data[i:]
                pose.body.confidence = pose.body.confidence[i:]
            break

    # Trim all trailing frames containing only zeros, almost no face or no hands
    for i in range(len(pose.body.data) - 1, 0, -1):
        if pose.body.confidence[i][:, 25:-42].sum() > face_th and \
                pose.body.confidence[i][:, 4] + pose.body.confidence[i][:, 7] > hands_th:
            if i != len(pose.body.data) - 1:
                pose.body.data = pose.body.data[:i + 1]
                pose.body.confidence = pose.body.confidence[:i + 1]
            break

    return pose


def process_datum(pose, flip=False):
    if flip:
        pose = flip_pose(pose)

    normalization_info = pose_normalization_info(pose.header)
    pose = pose.normalize(normalization_info)

    pose_hide_legs(pose)
    pose_hide_low_conf(pose)

    pose = trim_empty_frames(pose)

    return pose


def get_pose(keypoints_path, fps, flip):
    """
    Load OpenPose in the particular format (a single file for all frames).
    :param keypoints_path: Path to a folder that contains keypoints jsons (OpenPose output)
    for all frames of a video.
    :param fps: frame rate. default is 25.
    :return: Dictionary of Pose object with a header specific to OpenPose and a body that contains a
    single array.
    """
    files = sorted(os.listdir(keypoints_path))[:MAX_SEQ_LEN]
    frames = dict()
    for i, file in enumerate(files):
        try:
            with open(os.path.join(keypoints_path, file), "r") as openpose_raw:
                frame_json = json.load(openpose_raw)
                frames[i] = {"people": frame_json["people"][:1], "frame_id": i}
                cur_frame_pose = frame_json["people"][0]
                if (np.array(cur_frame_pose['pose_keypoints_2d'][7 * 3:7 * 3 + 2]) -
                    np.array(cur_frame_pose['hand_left_keypoints_2d'][0:2])).max() > 15 and \
                        cur_frame_pose['pose_keypoints_2d'][7 * 3 + 2] > MIN_CONFIDENCE:
                    cur_frame_pose['hand_left_keypoints_2d'][0:2] = cur_frame_pose['pose_keypoints_2d'][
                                                                    7 * 3:7 * 3 + 2]
                if (np.array(cur_frame_pose['pose_keypoints_2d'][4 * 3:4 * 3 + 2]) -
                    np.array(cur_frame_pose['hand_right_keypoints_2d'][0:2])).max() > 15 and \
                        cur_frame_pose['pose_keypoints_2d'][4 * 3 + 2] > MIN_CONFIDENCE:
                    cur_frame_pose['hand_right_keypoints_2d'][0:2] = cur_frame_pose['pose_keypoints_2d'][
                                                                     4 * 3:4 * 3 + 2]
        except:
            continue

    if len(frames) == 0:
        print(keypoints_path)
        return None

    # Convert to pose format
    pose = load_openpose(frames, fps=fps, width=400, height=400)
    pose = process_datum(pose, flip)
    return pose


def process_all_data(data, dup_keys, pjm_left_videos, keypoints_dir_path, fps=25):
    unique_dataset = []
    dup_dataset = []

    for key, val in data.items():
        if not os.path.isdir(os.path.join(keypoints_dir_path, key)):
            continue

        flip = key in pjm_left_videos
        pose = get_pose(os.path.join(keypoints_dir_path, key), fps, flip)
        if not pose:
            continue

        cur_instance = {"id": key,
                        "hamnosys": val["hamnosys"],
                        "text": val["type_name"],
                        "video": val["video_frontal"],
                        "fps": fps,
                        "pose": pose}

        if key in dup_keys:
            dup_dataset.append(cur_instance)
        else:
            unique_dataset.append(cur_instance)

    train_data, test_data = train_test_split(unique_dataset, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)
    train_data += dup_dataset

    with open("/mnt/raid1/home/rotem_shalev/motion-diffusion-model/dataset/ham2pose_processed_dataset_no_conf_th.pkl",
              'wb') as f:
        pkl.dump({"train": train_data, "val": val_data, "test": test_data}, f)


if __name__ == "__main__":
    # data = []
    # keypoints_path = "/mnt/raid1/home/rotem_shalev/how2sign/openpose_output/json"
    # dirs = os.listdir(keypoints_path)
    # for d in dirs:
    #     pose = get_pose(os.path.join(keypoints_path, d), 25, False)
    #     data.append({"id": d,
    #                  "hamnosys": "",
    #                  "text": "",
    #                  "video": "",
    #                  "fps": 25,
    #                  "pose": pose})
    # with open("/mnt/raid1/home/rotem_shalev/motion-diffusion-model/dataset/how2sign_processed_dataset_2.pkl",
    #           'wb') as f:
    #     pkl.dump(data, f)
    #
    # exit()

    data_dir_path = "/home/rotem_shalev/Ham2Pose/data/hamnosys"
    json_data_path = "/mnt/raid1/home/rotem_shalev/motion-diffusion-model/data_loaders/ham2pose/data.json"
    keypoints_dir_path = os.path.join(data_dir_path, "keypoints")

    pjm_left_videos_path = "/mnt/raid1/home/rotem_shalev/home/nlp/rotemsh/transcription/shared/pjm_left_videos.json"
    pose_header_path = os.path.join(data_dir_path, "openpose.poseheader")

    with open(pjm_left_videos_path, 'r') as f:
        pjm_left_videos = json.load(f)

    with open(json_data_path, 'r') as f:
        data = json.load(f)

    # signs with HamNoSys that appears more than once- must be in train!
    dup_keys = ['10248', '3516', '44909', '10573', '12916', '2674', '10753', '8044', '10890', '69225', '9280',
                '11286', '48575', '68699', '11288', '27428', '6248', '11291', '75271', '11420', '39949', '11435',
                '59785', '6230', '11874', '2294', '12278', '3071', '12641', '59684', '12844', '59701', '15121', '85192',
                '15286', '59212', '15735', '20652', '15962', '2803', '16153', '40233', '17265', '67630', '18003',
                '89436', '2442', '3048', '9028', '2452', '2856', '25235', '4511', '2686', '5035', '27521', '87394',
                '29817', '86689', '30365', '4171', '3172', '40005', '5908', '3193', '88457', '43516', '65542',
                '48749', '68018', '53036', '9386', '5492', '91376', '55848', '72736', '56000', '76667', '56684',
                '58318', '59424', '6192', '60848', '73060', '61731', '7247', '8291', '71120', '85160', '76557', '80774',
                '7940', '9790', '8265', '87255', '8289', '87848', 'FEUILLE', 'PAPIER', 'gsl_1024', 'gsl_165', 'gsl_124',
                'gsl_51', 'gsl_145', 'gsl_804', 'gsl_148', 'gsl_212', 'gsl_189', 'gsl_318', 'gsl_236', 'gsl_585',
                'gsl_244', 'gsl_965', 'gsl_27', 'gsl_504', 'gsl_339', 'gsl_530', 'gsl_353', 'gsl_719', 'gsl_424',
                'gsl_923', 'gsl_475', 'gsl_545', 'gsl_495', 'gsl_883', 'gsl_528', 'gsl_692']

    with open(pose_header_path, "rb") as buffer:
        pose_header = PoseHeader.read(BufferReader(buffer.read()))

    process_all_data(data, dup_keys, pjm_left_videos, keypoints_dir_path)
