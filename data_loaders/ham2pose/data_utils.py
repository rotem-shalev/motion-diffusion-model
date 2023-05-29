import requests
import os
import json
import pickle as pkl
from Levenshtein import distance
import numpy as np
import cv2
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.utils.reader import BufferReader
from pose_format.pose_header import PoseHeader
from data_loaders.ham2pose.preprocess_data import pose_hide_legs
from visualize.vis_2d import get_normalized_frames
from sample.generate import get_pred


def process_text(txt_path):
    # data = {}
    for srt_file in os.listdir(txt_path):
        print(srt_file)
        if os.path.isfile(f"meineDGS/processed_text/{srt_file[:-4]}.json"):
            continue
        with open(os.path.join(txt_path, srt_file), 'r') as f:
            lines = f.readlines()
        types_url = "https://www.sign-lang.uni-hamburg.de/meinedgs/ling/types_de.html"
        types_base_url = "https://www.sign-lang.uni-hamburg.de/meinedgs/"
        res = requests.get(types_url)
        types = res.text
        cur_data = []
        for i in range(len(lines)):
            if lines[i].isupper() and not "[" in lines[i]:
                time = lines[i-1].strip("\n")
                start, end = time.split(" --> ")
                signer_gloss = lines[i].strip("\n").strip("*").split(" ")
                if len(signer_gloss) != 2:
                    print(f"signer_gloss: {signer_gloss} in {srt_file}, i: {i}")
                    continue
                signer, gloss = signer_gloss
                entry = {"gloss": gloss, "signer": signer[:-1], "time": (start, end)}
                idx = types.find(">"+gloss)
                if idx != -1:
                    type_url = types[types[idx-100:idx].rfind("types/type")+idx-100:idx-1]
                    res = requests.get(types_base_url+type_url)
                    all_hamnosys = res.text[res.text.find('class="hamnosys"'):]
                    end_idx = all_hamnosys.find('>')
                    hamnosys_text = all_hamnosys[all_hamnosys.find('title="ham')+7:end_idx-1]
                    hamnosys = all_hamnosys[end_idx+1:all_hamnosys.find("</td>")]
                    entry["hamnosys_text"] = hamnosys_text
                    entry["hamnosys"] = hamnosys
                cur_data.append(entry)
        # data[srt_file[:-4]] = cur_data
        with open(f"meineDGS/processed_text/{srt_file[:-4]}.json", 'w', encoding='utf-8') as f:
            json.dump(cur_data, f)
    # with open("meineDGS/meineDGS_data.json", 'w', encoding='utf-8') as f:
    #     json.dump(data, f)


def get_meineDGS_data():
    os.makedirs("meineDGS/srt", exist_ok=True)
    os.makedirs("meineDGS/openpose", exist_ok=True)

    url = "https://www.sign-lang.uni-hamburg.de/meinedgs/ling/start-name_en.html"
    base_url = "https://www.sign-lang.uni-hamburg.de/meinedgs/"
    srt_base_url = base_url + "srt/"
    openpose_base_url = base_url + "openpose/"
    res = requests.get(url)
    t = res.text
    idx = t.find("<tr id=")

    while idx != -1:
        t = t[idx+1:]
        cur_entry = t[:t.find("</tr>")]
        id = cur_entry[7:cur_entry.find('">')]
        srt_url = srt_base_url+id+"_en.srt"
        srt_res = requests.get(srt_url)
        if srt_res.status_code == 200:
            if not os.path.isfile(f"meineDGS/srt/{id}.txt"):
                with open(f"meineDGS/srt/{id}.txt", 'w') as f:
                    f.write(srt_res.text)  # process text after saving
            if not os.path.isfile(f"meineDGS/openpose/{id}.json"):
                openpose_url = openpose_base_url+id+"_openpose.json.gz"
                openpose_res = requests.get(openpose_url)
                if srt_res.status_code == 200:
                    with open(f"meineDGS/openpose/{id}.json.gz", 'wb') as f:
                        f.write(openpose_res.content)
                    os.system(f"gunzip meineDGS/openpose/{id}.json.gz")
        idx = t.find("<tr id=")


def add_hamnosys_text(data_path):
    glyph2text = {' ': 'space', '!': 'exclam', ',': 'comma', '.': 'period', '?': 'question', '{': 'braceleft',
                  '|': 'bar', '}': 'braceright', '\xa0': 'space', '\ue000': 'hamfist', '\ue001': 'hamflathand',
                  '\ue002': 'hamfinger2', '\ue003': 'hamfinger23', '\ue004': 'hamfinger23spread', '\ue005': 'hamfinger2345',
                  '\ue006': 'hampinch12', '\ue007': 'hampinchall', '\ue008': 'hampinch12open', '\ue009': 'hamcee12',
                  '\ue00a': 'hamceeall', '\ue00b': 'hamceeopen', '\ue00c': 'hamthumboutmod', '\ue00d': 'hamthumbacrossmod',
                  '\ue00e': 'hamthumbopenmod', '\ue010': 'hamfingerstraightmod', '\ue011': 'hamfingerbendmod',
                  '\ue012': 'hamfingerhookmod', '\ue013': 'hamdoublebent', '\ue014': 'hamdoublehooked',
                  '\ue020': 'hamextfingeru', '\ue021': 'hamextfingerur', '\ue022': 'hamextfingerr', '\ue023': 'hamextfingerdr',
                  '\ue024': 'hamextfingerd', '\ue025': 'hamextfingerdl', '\ue026': 'hamextfingerl', '\ue027': 'hamextfingerul',
                  '\ue028': 'hamextfingerol', '\ue029': 'hamextfingero', '\ue02a': 'hamextfingeror', '\ue02b': 'hamextfingeril',
                  '\ue02c': 'hamextfingeri', '\ue02d': 'hamextfingerir', '\ue02e': 'hamextfingerui', '\ue02f': 'hamextfingerdi',
                  '\ue030': 'hamextfingerdo', '\ue031': 'hamextfingeruo', '\ue038': 'hampalmu', '\ue039': 'hampalmur',
                  '\ue03a': 'hampalmr', '\ue03b': 'hampalmdr', '\ue03c': 'hampalmud', '\ue03d': 'hampalmdl', '\ue03e': 'hampalml',
                  '\ue03f': 'hampalmul', '\ue040': 'hamhead', '\ue041': 'hamheadtop', '\ue042': 'hamforehead', '\ue043': 'hameyebrows',
                  '\ue044': 'hameyes', '\ue045': 'hamnose', '\ue046': 'hamnostrils', '\ue047': 'hamear', '\ue048': 'hamearlobe',
                  '\ue049': 'hamcheek', '\ue04a': 'hamlips', '\ue04b': 'hamtongue', '\ue04c': 'hamteeth', '\ue04d': 'hamchin',
                  '\ue04e': 'hamunderchin', '\ue04f': 'hamneck', '\ue050': 'hamshouldertop', '\ue051': 'hamshoulders',
                  '\ue052': 'hamchest', '\ue053': 'hamstomach', '\ue054': 'hambelowstomach', '\ue058': 'hamlrbeside', '\ue059': 'hamlrat',
                  '\ue05a': 'hamcoreftag', '\ue05b': 'hamcorefref', '\ue05f': 'hamneutralspace', '\ue060': 'hamupperarm',
                  '\ue061': 'hamelbow', '\ue062': 'hamelbowinside', '\ue063': 'hamlowerarm', '\ue064': 'hamwristback',
                  '\ue065': 'hamwristpulse', '\ue066': 'hamthumbball', '\ue067': 'hampalm', '\ue068': 'hamhandback',
                  '\ue069': 'hamthumbside', '\ue06a': 'hampinkyside', '\ue070': 'hamthumb', '\ue071': 'hamindexfinger',
                  '\ue072': 'hammiddlefinger', '\ue073': 'hamringfinger', '\ue074': 'hampinky', '\ue075': 'hamfingertip',
                  '\ue076': 'hamfingernail', '\ue077': 'hamfingerpad', '\ue078': 'hamfingermidjoint', '\ue079': 'hamfingerbase',
                  '\ue07a': 'hamfingerside', '\ue07c': 'hamwristtopulse', '\ue07d': 'hamwristtoback', '\ue07e': 'hamwristtothumb',
                  '\ue07f': 'hamwristtopinky', '\ue080': 'hammoveu', '\ue081': 'hammoveur', '\ue082': 'hammover', '\ue083': 'hammovedr',
                  '\ue084': 'hammoved', '\ue085': 'hammoveudl', '\ue086': 'hammovel', '\ue087': 'hammoveul', '\ue088': 'hammoveol',
                  '\ue089': 'hammoveo', '\ue08a': 'hammoveor', '\ue08b': 'hammoveil', '\ue08c': 'hammovei', '\ue08d': 'hammoveir',
                  '\ue08e': 'hammoveui', '\ue08f': 'hammovedi', '\ue090': 'hammovedo', '\ue091': 'hammoveuo',
                  '\ue092': 'hamcircleo', '\ue093': 'hamcirclei', '\ue094': 'hamcircled', '\ue095': 'hamcircleu', '\ue096': 'hamcirclel',
                  '\ue097': 'hamcircler', '\ue098': 'hamcircleul', '\ue099': 'hamcircledr', '\ue09a': 'hamcircleur',
                  '\ue09b': 'hamcircledl', '\ue09c': 'hamcircleol', '\ue09d': 'hamcircleir', '\ue09e': 'hamcircleor',
                  '\ue09f': 'hamcircleil', '\ue0a0': 'hamcircleui', '\ue0a1': 'hamcircledo', '\ue0a2': 'hamcircleuo',
                  '\ue0a3': 'hamcircledi', '\ue0a4': 'hamfingerplay', '\ue0a5': 'hamnodding', '\ue0a6': 'hamswinging',
                  '\ue0a7': 'hamtwisting', '\ue0a8': 'hamstircw', '\ue0a9': 'hamstirccw', '\ue0aa': 'hamreplace',
                  '\ue0ad': 'hammovecross', '\ue0ae': 'hammoveX', '\ue0af': 'hamnomotion', '\ue0b0': 'hamclocku',
                  '\ue0b1': 'hamclockul', '\ue0b2': 'hamclockl', '\ue0b3': 'hamclockdl', '\ue0b4': 'hamclockd',
                  '\ue0b5': 'hamclockdr', '\ue0b6': 'hamclockr', '\ue0b7': 'hamclockur', '\ue0b8': 'hamclockfull',
                  '\ue0b9': 'hamarcl', '\ue0ba': 'hamarcu', '\ue0bb': 'hamarcr', '\ue0bc': 'hamarcd', '\ue0bd': 'hamwavy',
                  '\ue0be': 'hamzigzag', '\ue0c0': 'hamellipseh', '\ue0c1': 'hamellipseur', '\ue0c2': 'hamellipsev',
                  '\ue0c3': 'hamellipseul', '\ue0c4': 'hamincreasing', '\ue0c5': 'hamdecreasing', '\ue0c6': 'hamsmallmod',
                  '\ue0c7': 'hamlargemod', '\ue0c8': 'hamfast', '\ue0c9': 'hamslow', '\ue0ca': 'hamtense', '\ue0cb': 'hamrest',
                  '\ue0cc': 'hamhalt', '\ue0d0': 'hamclose', '\ue0d1': 'hamtouch', '\ue0d2': 'haminterlock', '\ue0d3': 'hamcross',
                  '\ue0d4': 'hamarmextended', '\ue0d5': 'hambehind', '\ue0d6': 'hambrushing', '\ue0d8': 'hamrepeatfromstart',
                  '\ue0d9': 'hamrepeatfromstartseveral', '\ue0da': 'hamrepeatcontinue', '\ue0db': 'hamrepeatcontinueseveral',
                  '\ue0dc': 'hamrepeatreverse', '\ue0dd': 'hamalternatingmotion', '\ue0e0': 'hamseqbegin', '\ue0e1': 'hamseqend',
                  '\ue0e2': 'hamparbegin', '\ue0e3': 'hamparend', '\ue0e4': 'hamfusionbegin', '\ue0e5': 'hamfusionend',
                  '\ue0e6': 'hambetween', '\ue0e7': 'hamplus', '\ue0e8': 'hamsymmpar', '\ue0e9': 'hamsymmlr',
                  '\ue0ea': 'hamnondominant', '\ue0eb': 'hamnonipsi', '\ue0ec': 'hametc', '\ue0ed': 'hamorirelative',
                  '\ue0f0': 'hammime', '\ue0f1': 'hamversion40'}

    with open(data_path, 'r') as f:
        data_json = json.load(f)

    for d in data_json.values():
        glyphs = list(d["hamnosys"])
        text = d["hamnosys_text"].split(",") if d["hamnosys_text"] else []
        if len(glyphs) != len(text):
            d["hamnosys_text"] = ",".join([glyph2text[glyph] for glyph in glyphs])

    with open("data.json", 'w') as f:
        json.dump(data_json, f, indent=4)


def visualize_poses(ref, similar_signs, fname):
    all_frames = get_normalized_frames([ref]+[sign[1] for sign in similar_signs])
    text_margin = 50
    font = cv2.FONT_HERSHEY_TRIPLEX
    color = (0, 0, 0)
    fps = 25
    labels = ["ref"] + [f"{sign[0]}_{sign[2]}" for sign in similar_signs]
    w = max([frames[0].shape[1] for frames in all_frames])
    h = max([frames[0].shape[0] for frames in all_frames])
    image_size = (w * len(labels), h + text_margin)
    out = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, image_size)
    max_len = max([len(frames) for frames in all_frames])

    for i in range(max_len+1):
        all_video_frames = []
        for j, frames in enumerate(all_frames):
            if len(frames) > i:
                cur_frame = np.full((image_size[1], image_size[0] // len(labels), 3), 255, dtype=np.uint8)
                cur_frame[text_margin:frames[i].shape[0] + text_margin, :frames[i].shape[1]] = frames[i]
                cur_frame = cv2.putText(cur_frame, labels[j], (5, 20), font, 0.5, color, 1, 0)
            else:
                cur_frame = prev_video_frames[j]
            all_video_frames.append(cur_frame)
        merged_frame = np.concatenate(all_video_frames, axis=1)
        out.write(merged_frame)
        prev_video_frames = all_video_frames

    out.release()


def convert_pred_to_pose(seq):
    pose_header_path = "/home/rotem_shalev/Ham2Pose/data/hamnosys/openpose.poseheader"
    with open(pose_header_path, "rb") as buffer:
        pose_header = PoseHeader.read(BufferReader(buffer.read()))

    data = seq.reshape(-1, 1, seq.shape[1], seq.shape[2])
    conf = np.ones_like(data[:, :, :, 0])
    pose_body = NumPyPoseBody(25, data, conf)
    predicted_pose = Pose(pose_header, pose_body)
    pose_hide_legs(predicted_pose)
    return predicted_pose


def get_text_embedding(txt):
    import torch
    from utils.fixseed import fixseed
    from utils.parser_util import generate_args
    from utils import dist_util
    from sample.generate import load_dataset
    from utils.model_util import create_model_and_diffusion, load_model_wo_clip

    args = generate_args()
    fixseed(args.seed)
    max_frames = 200
    n_frames = max_frames
    dist_util.setup_dist(args.device)

    data = load_dataset(args, max_frames, n_frames, split=args.split)
    model, diffusion = create_model_and_diffusion(args, data)
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)
    model.to(dist_util.dev())
    model.eval()
    return model.encode_text(txt)[0].transpose(0, 1).cpu().detach().numpy()


def pad(vec, size):
    if len(vec) >= size:
        return vec.reshape(-1, 1)
    return np.concatenate([vec, np.zeros(size-len(vec))]).reshape(-1, 1)


def get_k_most_similar(txt, k, id="", method="text", split="test"):
    with open("/mnt/raid1/home/rotem_shalev/motion-diffusion-model/dataset/ham2pose_processed_dataset_2.pkl",
              'rb') as f:
        data = pkl.load(f)

    length = 200  # max_frames
    fname = f"eval/ham2pose/videos/{k}_most_similar.mp4"

    if id:
        length = [len(d["pose"].body.data) for d in data[split] if d["id"] == id][0]
        fname = f"eval/ham2pose/videos/{id}_{k}_most_similar.mp4"
    pred = convert_pred_to_pose(get_pred(txt, length))

    all_train_data = [(d["hamnosys"], d["id"]) for d in data["train"]]
    if method == "text":
        id2dist = {d[1]: distance(txt, d[0]) for d in all_train_data}
    elif method == "embedding":
        from sklearn.metrics.pairwise import cosine_similarity
        ref_embedding = get_text_embedding([txt]).squeeze().sum(axis=0).reshape(1, -1)
        batch_size = 32
        id2embedding = {}
        for i in range(0, batch_size, len(all_train_data)):
            ids = [d[1] for d in all_train_data[i:i+batch_size]]
            texts = [d[0] for d in all_train_data[i:i+batch_size]]
            embeddings = get_text_embedding(texts)
            id2embedding.update(dict(zip(ids, embeddings)))

        id2dist = {id: abs(cosine_similarity(ref_embedding, emb.sum(axis=0).reshape(1, -1))[0][0])
                   for id, emb in id2embedding.items()}

    id2dist_sorted = sorted(id2dist.items(), key=lambda x: x[1])
    k_most_similar_ids = [s[0] for s in id2dist_sorted[:k]]
    similar_signs = [(d["id"], d["pose"], id2dist[d["id"]]) for d in data["train"] if d["id"] in k_most_similar_ids]
    visualize_poses(pred, similar_signs, fname)


if __name__ == "__main__":
    # get_meineDGS_data()
    # process_text("/mnt/raid1/home/rotem_shalev/motion-diffusion-model/data_loaders/ham2pose/meineDGS/srt")
    # add_hamnosys_text("/home/rotem_shalev/Ham2Pose/data/hamnosys/data.json")
    txt = "\ue005\ue00e\ue020\ue03e\ue049\ue059\ue0e0\ue0d1\ue075\ue0e1\ue0e2\ue083\ue0aa\ue053\ue007\ue010\ue028\ue03c" \
          "\ue0e3"
    id = "pjm_1189"
    get_k_most_similar(txt, k=5, id=id, method="text", split="test")
