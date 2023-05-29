import numpy as np
import torch
from utils.fixseed import fixseed
from data_loaders.get_data import get_dataset_loader
from utils.parser_util import evaluation_parser
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util


def main():
    args = evaluation_parser()
    fixseed(42)
    split = 'test'
    n_frames = 60#40

    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=n_frames,
                              split=split,
                              hml_mode='text_only')
    mu, sigma = get_stats(data)
    print(f"GT mu: {mu}, sigma: {sigma}")

    model, diffusion = create_model_and_diffusion(args, data)
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)
    model.to(dist_util.dev())
    model.eval()

    iterator = iter(data)
    _, model_kwargs = next(iterator)
    rot2xyz_pose_rep = model.data_rep
    all_motions = []

    sample_fn = diffusion.p_sample_loop

    rot2xyz_mask = model_kwargs['y']['mask'].reshape(args.batch_size, n_frames).bool()
    # for _ in iter(data):
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
    cur_sample = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True,
                               translation=True, jointstype='smpl', vertstrans=True, betas=None, beta=0,
                               glob_rot=None, get_rotations_back=False)
    all_motions.append(cur_sample.cpu().numpy())

    # for i, motions in enumerate(all_motions):
    #     cur_mu, cur_sigma = get_stats(motions)
    #     print(f"Predicted {i} mu: {cur_mu}, sigma: {cur_sigma}")

    generated_motions = np.concatenate(all_motions, axis=0)

    g_mu, g_sigma = get_stats(generated_motions)
    print(f"Predicted all mu: {g_mu}, sigma: {g_sigma}")

    print(f"abs diff mu: {abs(mu-g_mu)}, sigma: {abs(sigma-g_sigma)}")


def get_stats(data):
    if isinstance(data, np.ndarray):
        return np.mean(data), np.std(data)
    data_list = [d[0].numpy()[:, :-1, :, :] for d in iter(data)]
    mu = np.mean(data_list)
    sigma = np.std(data_list)
    return mu, sigma


if __name__ == "__main__":
    main()
