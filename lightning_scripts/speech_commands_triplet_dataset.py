"""
Speech Commands level-discrimination triplet dataset.

Extracted from notebooks/zero_shot_speech_commands_level_discrimination.ipynb
so it can be imported by standalone evaluation scripts.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class SpeechCommandsLevelTripletDataset(torch.utils.data.Dataset):
    """
    Triplet dataset for level discrimination on Speech Commands.

    Each clip concatenates two spoken words at different loudness levels.
    - Anchor:   word1_A + word2_A with a level relationship
    - Positive: word1_B + word2_B with the **same** level relationship
    - Negative: word1_B + word2_B with the **reversed** level relationship
    """

    def __init__(self, dataset_split, seed=0, num_triplets=2000,
                 target_sr=20_000, sig_length=40_000):
        self.dataset_split = dataset_split
        self.target_sr = target_sr
        self.sample_rate = target_sr
        self.sig_length = sig_length

        self.label_list = sorted({ex["label"] for ex in dataset_split})
        self.examples_by_label = {}
        for idx, ex in enumerate(dataset_split):
            self.examples_by_label.setdefault(ex["label"], []).append(idx)

        rng = np.random.RandomState(seed)
        self.triplets = []
        for _ in range(num_triplets):
            w1, w2 = rng.choice(self.label_list, size=2, replace=True)
            w1_higher = bool(rng.choice([True, False]))
            self.triplets.append((w1, w2, w1_higher))

    def __len__(self):
        return len(self.triplets)

    def _load_word(self, idx):
        ex = self.dataset_split[int(idx)]
        audio = torch.from_numpy(ex["audio"]["array"]).float()
        sr = ex["audio"]["sampling_rate"]
        if sr != self.target_sr:
            audio = torchaudio.functional.resample(
                audio.unsqueeze(0), sr, self.target_sr
            ).squeeze(0)
        rms = torch.sqrt(torch.mean(audio ** 2))
        if rms > 1e-6:
            audio = audio / rms
        return audio

    def _make_sequence(self, w1_label, w2_label, w1_idx, w2_idx, w1_higher):
        a1 = self._load_word(w1_idx)
        a2 = self._load_word(w2_idx)
        level_ratio = 0.20
        if w1_higher:
            a2 = a2 * level_ratio
        else:
            a1 = a1 * level_ratio
        seq = torch.cat([a1, a2])
        if len(seq) < self.sig_length:
            pad = (self.sig_length - len(seq)) // 2 + 1
            seq = F.pad(seq, (pad, pad))
        start = (len(seq) - self.sig_length) // 2
        return seq[start:start + self.sig_length]

    def __getitem__(self, idx):
        w1, w2, w1_higher = self.triplets[idx]
        rng = np.random.RandomState(idx)

        available = [w for w in self.label_list if w != w1 and w != w2]
        if len(available) < 4:
            available = list(self.label_list)

        w1_B, w2_B = rng.choice(available, size=2, replace=False)

        idx_a1 = rng.choice(self.examples_by_label[w1])
        idx_a2 = rng.choice(self.examples_by_label[w2])
        idx_b1 = rng.choice(self.examples_by_label[w1_B])
        idx_b2 = rng.choice(self.examples_by_label[w2_B])

        anchor = self._make_sequence(w1, w2, idx_a1, idx_a2, w1_higher)
        positive = self._make_sequence(w1_B, w2_B, idx_b1, idx_b2, w1_higher)
        negative = self._make_sequence(w1_B, w2_B, idx_b1, idx_b2, not w1_higher)

        clips = {"anchor": anchor, "positive": positive, "negative": negative}
        triplet_info = {
            "anchor": {"word1": w1, "word2": w2, "word1_higher": w1_higher},
            "positive": {"word1": w1_B, "word2": w2_B, "word1_higher": w1_higher},
            "negative": {"word1": w1_B, "word2": w2_B, "word1_higher": not w1_higher},
        }
        return clips, self.target_sr, triplet_info
