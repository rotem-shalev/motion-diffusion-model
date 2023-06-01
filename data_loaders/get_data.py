from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate
from math import inf


def get_dataset_class(name):
    if name == "amass":
        from data_loaders.amass import AMASS
        return AMASS
    elif name == "uestc":
        from data_loaders.a2m.uestc import UESTC
        return UESTC
    elif name == "humanact12":
        from data_loaders.a2m.humanact12poses import HumanAct12Poses
        return HumanAct12Poses
    elif name == "humanml":
        from data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == "kit":
        from data_loaders.humanml.data.dataset import KIT
        return KIT
    elif name == "interhand":
        from data_loaders.interhand.dataset import InterHand
        return InterHand
    elif name == "hanco":
        from data_loaders.hanco.dataset import HanCO
        return HanCO
    elif name == "grab":
        from data_loaders.grab.dataset import GRAB
        return GRAB
    elif name == "hoi4d":
        from data_loaders.hoi4d.dataset import HOI4D
        return HOI4D
    elif name == "all_hands":
        from data_loaders.all_hands.dataset import ALL_HANDS
        return ALL_HANDS
    elif name == "ham2pose":
        from data_loaders.ham2pose.dataset import Ham2Pose
        return Ham2Pose
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')


def get_collate_fn(name, hml_mode='train'):
    if hml_mode == 'gt':
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name in ["humanml", "kit"]:
        return t2m_collate
    else:
        return all_collate


def get_dataset(name, num_frames, split='train', hml_mode='train', max_seq_num=inf, subset="", use_how2sign=False,
                conf_power=1.0):
    DATA = get_dataset_class(name)
    if name in ["humanml", "kit"]:
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode)
    elif name == "all_hands":
        dataset = DATA(split=split, num_frames=num_frames, max_seq_num=max_seq_num, subset=subset)
    elif name == "ham2pose":
        dataset = DATA(split=split, num_frames=num_frames, max_seq_num=max_seq_num, use_how2sign=use_how2sign,
                       conf_power=conf_power)
    else:
        dataset = DATA(split=split, num_frames=num_frames, max_seq_num=max_seq_num)
    return dataset


def get_dataset_loader(name, batch_size, num_frames, split='train', hml_mode='train', max_seq_num=inf, subset="",
                       use_how2sign=False, conf_power=1.0):
    dataset = get_dataset(name, num_frames, split, hml_mode, max_seq_num=max_seq_num, subset=subset,
                          use_how2sign=use_how2sign, conf_power=conf_power)
    collate = get_collate_fn(name, hml_mode)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=split == 'train',
        num_workers=0, drop_last=True, collate_fn=collate
    )

    return loader


# data = get_dataset("humanml", 60)
# for d in data:
#     print(d)
# get_dataset_loader("humanact12", 64, 60)
# get_dataset_loader("interhand", 64, 60, split="test")