from pathlib import Path
from typing import List
import torch
from fontTools.ttLib import TTFont

from data_loaders.ham2pose.collator import zero_pad_collator


class HamNoSysTokenizer:

    def __init__(self, split_repeat=False):
        self.split_repeat = split_repeat
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2

        circle_tokens = {'\ue092': 'o', '\ue093': 'i', '\ue094': 'd', '\ue095': 'u', '\ue096': 'l',
                  '\ue097': 'r', '\ue098': 'ul', '\ue099': 'dr', '\ue09a': 'ur',
                  '\ue09b': 'dl', '\ue09c': 'ol', '\ue09d': 'ir', '\ue09e': 'or',
                  '\ue09f': 'il', '\ue0a0': 'ui', '\ue0a1': 'do', '\ue0a2': 'uo',
                  '\ue0a3': 'di'}

        self.font_path = Path(__file__).parent.joinpath("HamNoSysUnicode.ttf")

        with TTFont(self.font_path) as font:
            tokens = [chr(key) for key in font["cmap"].getBestCmap().keys()]
            # print({chr(key): val for key, val in font["cmap"].getBestCmap().items()})

        self.i2s = {(i + 3): c for i, c in enumerate(tokens)}
        self.s2i = {c: i for i, c in self.i2s.items()}

    def __len__(self):
        return len(self.i2s) + 3

    def tokenize(self, text: str):
        if self.split_repeat:
            hamrepeatfromstart = "\ue0d8"
            hamreplace = "\ue0aa"
            idx = text.find(hamrepeatfromstart)
            if idx != -1:
                text = text[:idx] + hamreplace + text[:idx] + text[idx+1:]  # TODO- what if repeat is not over all the
                # sequence?

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
    tokenizer = HamNoSysTokenizer()
    hamnosys = [
        "\ue002\ue0e6\ue002\ue010\ue027\ue03e\ue052\ue0d0\ue093\ue0d8"
        # "",  # bsl one
        # "",  # gsl one
        # "\ue000\ue071",
        # "\ue000\ue071\ue012\ue029\ue03f\ue089\ue0c6\ue0d8 \ue000\ue071"
    ]
    print(hamnosys)
    print(tokenizer(hamnosys))
