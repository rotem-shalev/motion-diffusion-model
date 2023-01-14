import os

rootdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SMPL_DATA_PATH = os.path.join(rootdir, "body_models/smpl")

SMPL_KINTREE_PATH = os.path.join(SMPL_DATA_PATH, "kintree_table.pkl")
SMPL_MODEL_PATH = os.path.join(SMPL_DATA_PATH, "SMPL_NEUTRAL.pkl")
SMPL_MANO_MODEL_PATH = os.path.join(SMPL_DATA_PATH, "MANO_RIGHT.pkl")  # TODO- currently only right hand
JOINT_REGRESSOR_TRAIN_EXTRA = os.path.join(SMPL_DATA_PATH, 'J_regressor_extra.npy')

ROT_CONVENTION_TO_ROT_NUMBER = {
    'legacy': 23,
    'no_hands': 21,
    'full_hands': 51,
    'mitten_hands': 33,
}

GENDERS = ['neutral', 'male', 'female']
NUM_BETAS = 10