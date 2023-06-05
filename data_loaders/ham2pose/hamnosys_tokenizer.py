from pathlib import Path
from typing import List
import torch
from fontTools.ttLib import TTFont

from data_loaders.ham2pose.collator import zero_pad_collator


class HamNoSysTokenizer:

    def __init__(self, split_repeat=False, split_move_direction=False):
        self.split_repeat = split_repeat
        self.split_move_direction = split_move_direction

        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.num_special_tokens = 3

        # TODO- same for move, palm, extfinger? what about finger/thumb roles? only 2-3 for each of them

        self.font_path = Path(__file__).parent.joinpath("HamNoSysUnicode.ttf")

        with TTFont(self.font_path) as font:
            tokens = [chr(key) for key in font["cmap"].getBestCmap().keys()]
            ham2token = {val: chr(key) for key, val in font["cmap"].getBestCmap().items()}

        split_ham = {}
        suffix_tokens = {}

        if self.split_repeat:
            split_ham.update({"repeatfrom": '\ue0d8'})
            suffix_tokens.update({"start": '2', "startseveral": '3'})  # TODO- add hamrepeatreverse too?

        if self.split_move_direction:
            split_ham.update({"circle": 'c', "move": 'm'})
            # split_ham.update({"circle": '\ue092', "move": '\ue081', "symm": '\ue0e9',
            #              "extfinger": '\ue020'})  # TODO- uncomment for symm_extfinger

            suffix_tokens.update({'o': 'o', 'i': 'i', 'd': 'd', 'u': 'u', 'l': 'l', 'r': 'r', 'ul': 'a', 'dr': 'b',
                                'ur': 'n', 'dl': 'q', 'ol': 'e', 'ir': 'f', 'or': 'g', 'il': 'h', 'ui': 'p',
                                # TODO- 'ui': 'p' for move_direction, 'ui': 'w' for symm_extfinger
                                'do': 'j', 'uo': 'k', 'di': 's', 'udl': 't', 'X': 'X', 'cross': 'x',
                                # 'lr': 'z', 'par': 'p' # TODO- uncomment for symm_extfinger
                                })

            assert len(set(suffix_tokens.values())) == len(suffix_tokens.values())

        if split_ham:
            tokens += list(split_ham.values())
            tokens += list(suffix_tokens.values())

            self.split_tokens = {}
            for h in ham2token:
                for ham in split_ham:
                    if ham in h:
                        tokens.remove(ham2token[h])
                        self.split_tokens[ham2token[h]] = split_ham[ham] + suffix_tokens[h[len(f'ham{ham}'):]]
                        break

        self.i2s = {(i + self.num_special_tokens): c for i, c in enumerate(tokens)}
        self.s2i = {c: i for i, c in self.i2s.items()}

    def __len__(self):
        return len(self.i2s) + self.num_special_tokens

    def tokenize(self, text: str):
        # if self.split_repeat:
        #     hamrepeatfromstart = "\ue0d8"
        #     hamreplace = "\ue0aa"
        #     idx = text.find(hamrepeatfromstart)
        #     if idx != -1:
        #         text = text[:idx] + hamreplace + text[:idx] + text[idx+1:]  # TODO- what if repeat is not over all the
                # sequence?
        if self.split_tokens:
            for token in self.split_tokens:
                text = text.replace(token, self.split_tokens[token])

        return [self.bos_token_id] + [self.s2i[c] for c in text] + [self.eos_token_id]

    def __call__(self, texts: List[str], device=None):
        all_tokens = [self.tokenize(text) for text in texts]

        tokens_batch = zero_pad_collator([{
            "tokens_ids": torch.tensor(tokens, dtype=torch.long, device=device),
            "attention_mask": torch.ones(len(tokens), dtype=torch.bool, device=device),
            "positions": torch.arange(0, len(tokens), dtype=torch.long, device=device)
        } for tokens in all_tokens])
        # In transformers, 1 is mask, not 0
        tokens_batch["attention_mask"] = torch.logical_not(tokens_batch["attention_mask"])

        return tokens_batch


if __name__ == "__main__":
    tokenizer = HamNoSysTokenizer(split_move_direction=True)
    hamnosys = [
        "\ue002\ue0e6\ue002\ue010\ue027\ue03e\ue052\ue0d0\ue093\ue0d8"
        # "",  # bsl one
        # "",  # gsl one
        # "\ue000\ue071",
        # "\ue000\ue071\ue012\ue029\ue03f\ue089\ue0c6\ue0d8 \ue000\ue071"
    ]
    print(hamnosys)
    print(tokenizer(hamnosys))
