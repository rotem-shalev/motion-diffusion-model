import os
import sys

rootdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, rootdir)

import numpy as np
from numpy import ma
import torch
from pose_format.utils.reader import BufferReader
from pose_format.pose_header import PoseHeader
from utils.fixseed import fixseed
from utils.parser_util import generate_args
from data_loaders.get_data import get_dataset_loader
from model.cfg_sampler import ClassifierFreeSampleModel
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

MIN_CONFIDENCE = 0.2


def unmask_pose(masked_pose, other_masked_pose):
    if not ma.is_masked(masked_pose):
        return np.array(masked_pose).reshape(-1, 51*2)
    mask = ma.getmask(masked_pose)
    mask2 = ma.getmask(other_masked_pose)
    new_pose = []

    for i in range(len(mask)):
        new_pose.append([])
        for j in range(len(mask[i])):
            if not mask[i][j][0]:
                new_pose[i] += list(np.array(masked_pose[i][j]))
            elif i >= len(other_masked_pose) or (ma.is_masked(other_masked_pose) and mask2[i][j][0]):
                new_pose[i] += [0, 0]
            else:
                new_pose[i] += list(np.array(other_masked_pose[i][j]) / 2)

    return np.array(new_pose)


def seq_dist(pose1, pose2):
    return np.linalg.norm(pose1-pose2)


def DTW_MJE(poses_data, ind):
    masked_pose1 = poses_data[0][:, ind]
    masked_pose2 = poses_data[1][:, ind]
    new_pose_1 = unmask_pose(masked_pose1, masked_pose2)
    new_pose_2 = unmask_pose(masked_pose2, masked_pose1)
    dist, path = fastdtw(new_pose_1, new_pose_2, dist=seq_dist)

    mje_dist = 0
    for tup in path:
        mje_dist += np.linalg.norm(new_pose_1[tup[0]]-new_pose_2[tup[1]])

    return mje_dist


def masked_euclidean(point1, point2):
    if np.ma.is_masked(point2):  # reference label keypoint is missing
        return 0
    elif np.ma.is_masked(point1):  # reference label keypoint is not missing, other label keypoint is missing
        return euclidean((0, 0), point2)/2
    d = euclidean(point1, point2)
    return d


def get_keys(max_idx):
    # return relevant keys- don't use legs, face for trajectory distance computations- only upper body and hands
    keys = list(range(9))
    keys += list(range(95, max_idx))
    return keys


def compare_poses(pose1, pose2):
    poses_data = [pose1, pose2]
    keys = get_keys(pose1.shape[1])
    dtw_mje = DTW_MJE(poses_data, keys)
    return dtw_mje


def check_ranks(distances, index):
    rank_1 = (index == distances[0])
    rank_5 = (index in distances[:5])
    rank_10 = (index in distances)
    return rank_1, rank_5, rank_10


def mask_pose(pose, confidence):
    mask = confidence <= MIN_CONFIDENCE
    stacked_confidence = np.stack([mask, mask], axis=2)
    masked_data = ma.masked_array(pose, mask=stacked_confidence)
    return masked_data


def calc_all_distances(preds, gts, confs):
    n = len(preds)
    pred_to_gt = np.zeros((n, n))
    pred_to_pred = np.zeros((n, n))
    gt_to_gt = np.zeros((n, n))
    gt_to_pred = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            masked_gt = mask_pose(gts[j], confs[j])
            pred_to_gt[j, i] = compare_poses(preds[i], masked_gt)
            pred_to_pred[j, i] = compare_poses(preds[i], preds[j])
            gt_to_gt[j, i] = compare_poses(mask_pose(gts[i], confs[i]), masked_gt)
            gt_to_pred[j, i] = compare_poses(masked_gt, preds[i])

    return pred_to_gt, pred_to_pred, gt_to_gt, gt_to_pred


def get_poses_ranks(cur_i, pred_to_gt_dist, pred_to_pred_dist, gt_to_gt_dist, gt_to_pred_dist):
    pred2label_distance = pred_to_gt_dist[cur_i, cur_i]
    distances_to_label = np.concatenate([pred_to_gt_dist[cur_i], gt_to_gt_dist[cur_i][:cur_i],
                                         gt_to_gt_dist[cur_i][cur_i+1:]])

    distances_to_pred = np.concatenate([gt_to_pred_dist[cur_i], pred_to_pred_dist[cur_i][:cur_i],
                                        pred_to_pred_dist[cur_i][cur_i+1:]])

    best_pred = np.argsort(distances_to_pred)[:10]
    rank_1_pred, rank_5_pred, rank_10_pred = check_ranks(best_pred, cur_i)
    best_label = np.argsort(distances_to_label)[:10]
    rank_1_label, rank_5_label, rank_10_label = check_ranks(best_label, cur_i)

    return pred2label_distance, rank_1_pred, rank_5_pred, rank_10_pred, rank_1_label, rank_5_label, rank_10_label


def test_distance_ranks(preds, gts, confs, ids):
    with torch.no_grad():
        rank_1_pred_sum = rank_5_pred_sum = rank_10_pred_sum = rank_1_label_sum = rank_5_label_sum = \
            rank_10_label_sum = 0
        pred2label_distances = dict()
        pred_to_gt_dist, pred_to_pred_dist, gt_to_gt_dist, gt_to_pred_dist = calc_all_distances(preds, gts, confs)
        for i in range(len(preds)):
            pred2label_distance, rank_1_pred, rank_5_pred, rank_10_pred, rank_1_label, rank_5_label, \
            rank_10_label = get_poses_ranks(i, pred_to_gt_dist, pred_to_pred_dist, gt_to_gt_dist, gt_to_pred_dist)
            pred2label_distances[ids[i]] = pred2label_distance
            print(f"{ids[i]} ranks:\n "
                  f"{pred2label_distance}, {rank_1_pred}, {rank_5_pred}, {rank_10_pred}, {rank_1_label},"
                  f" {rank_5_label}, {rank_10_label}")
            rank_1_pred_sum += int(rank_1_pred)
            rank_5_pred_sum += int(rank_5_pred)
            rank_10_pred_sum += int(rank_10_pred)
            rank_1_label_sum += int(rank_1_label)
            rank_5_label_sum += int(rank_5_label)
            rank_10_label_sum += int(rank_10_label)

        return rank_1_pred_sum, rank_5_pred_sum, rank_10_pred_sum, rank_1_label_sum, rank_5_label_sum, rank_10_label_sum


if __name__ == "__main__":
    args = generate_args()
    fixseed(args.seed)

    name = os.path.basename(os.path.dirname(args.model_path))
    print(f"evaluating {name}")
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 200
    fps = 25
    n_frames = max_frames

    dist_util.setup_dist(args.device)

    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split="test",
                              hml_mode="text_only",
                              subset=None,
                              max_seq_num=args.max_seq_num)

    data.fixed_length = n_frames

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    iterator = iter(data)

    sample_fn = diffusion.p_sample_loop

    pose_header_path = "visualize/openpose.poseheader"
    with open(pose_header_path, "rb") as buffer:
        pose_header = PoseHeader.read(BufferReader(buffer.read()))

    rank_1_pred_sum = rank_5_pred_sum = rank_10_pred_sum = rank_1_label_sum = rank_5_label_sum = rank_10_label_sum = 0

    for batch in iterator:
        gt_motion, model_kwargs = batch
        # add CFG scale to batch
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

        if 'mask' in model_kwargs['y']:
            padding_mask = torch.zeros((args.batch_size, 1, 1, max_frames-model_kwargs['y']['mask'].shape[-1]), dtype=torch.bool)
            model_kwargs['y']['mask'] = torch.cat([model_kwargs['y']['mask'], padding_mask], dim=-1).to(dist_util.dev())

        sample = sample_fn(
            model,
            (args.batch_size, model.njoints, model.nfeats, n_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        rot2xyz_pose_rep = 'xyz'
        rot2xyz_mask = None

        sample = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                               jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                               get_rotations_back=False)

        all_ids = model_kwargs['y']['id']
        all_lengths = model_kwargs['y']['lengths'].cpu().numpy()
        all_motions = sample.cpu().numpy()
        all_gts = gt_motion['motion'].cpu().numpy()
        all_confs = gt_motion['confidence'].cpu().numpy()

        all_motions = np.array([all_motions[i].transpose(2, 0, 1)[:all_lengths[i]] for i in range(len(all_motions))])
        all_gts = np.array([all_gts[i].transpose(2, 0, 1)[:all_lengths[i]] for i in range(len(all_gts))])
        all_confs = np.array([all_confs[i].transpose(1, 0)[:all_lengths[i]] for i in range(len(all_confs))])

        cur_rank_1_pred_sum, cur_rank_5_pred_sum, cur_rank_10_pred_sum, cur_rank_1_label_sum, cur_rank_5_label_sum, \
        cur_rank_10_label_sum = test_distance_ranks(all_motions, all_gts, all_confs, all_ids)

        rank_1_pred_sum += cur_rank_1_pred_sum
        rank_5_pred_sum += cur_rank_5_pred_sum
        rank_10_pred_sum += cur_rank_10_pred_sum
        rank_1_label_sum += cur_rank_1_label_sum
        rank_5_label_sum += cur_rank_5_label_sum
        rank_10_label_sum += cur_rank_10_label_sum

    num_samples = len(data.dataset)

    print(f"rank 1 pred sum: {rank_1_pred_sum} / {num_samples}: {rank_1_pred_sum / num_samples}")
    print(f"rank 5 pred sum: {rank_5_pred_sum} / {num_samples}: {rank_5_pred_sum / num_samples}")
    print(f"rank 10 pred sum: {rank_10_pred_sum} / {num_samples}: {rank_10_pred_sum / num_samples}")

    print(f"rank 1 label sum: {rank_1_label_sum} / {num_samples}: {rank_1_label_sum / num_samples}")
    print(f"rank 5 label sum: {rank_5_label_sum} / {num_samples}: {rank_5_label_sum / num_samples}")
    print(f"rank 10 label sum: {rank_10_label_sum} / {num_samples}: {rank_10_label_sum / num_samples}")
