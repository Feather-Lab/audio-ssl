"""Self-contained Tone Perfect triplet dataset for Mandarin tone discrimination."""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torchaudio
from torchaudio.transforms import Resample

import robustness.audio_functions.audio_transforms as at


def load_audio(rel_path, tone_perfect_dir):
    """Load audio file from Tone Perfect dataset."""
    audio_path = tone_perfect_dir / rel_path
    audio, sr = torchaudio.load(str(audio_path))
    return audio, sr


class TonePerfectDataset(torch.utils.data.Dataset):
    """Self-contained Tone Perfect triplet generator for Mandarin tone discrimination.
    
    Generates balanced triplets across tones where:
    - Anchor and positive: same speaker, same tone, different base syllables
    - Negative: same speaker, different tone
    """

    def __init__(
        self,
        tone_perfect_dir,
        resample_sr=20_000,
        pair_mode=False,
        include_negative=True,
        allow_tone1=False,
        n_examples=None,
        random_seed=42,
    ):
        super().__init__()
        tone_perfect_SR = 44100
        
        tp_metadata = pd.read_csv(tone_perfect_dir / "audio_metadata.csv")
        if not allow_tone1:
            tp_metadata = tp_metadata[tp_metadata.tone != 1]
        self.metadata = tp_metadata
        self.talker_ids = tp_metadata.speaker.unique()
        self.base_syllables = tp_metadata.base_syllable.unique()
        self.tones = sorted(tp_metadata.tone.unique())
        self.pair_mode = pair_mode
        self.include_negative = include_negative
        self.n_examples = n_examples
        self.random_seed = random_seed
        self.resample_sr = resample_sr
        self.tone_perfect_dir = Path(tone_perfect_dir)
        self.tone_perfect_SR = tone_perfect_SR
        
        if resample_sr is not None:
            self.resamp = Resample(tone_perfect_SR, resample_sr)
        else:
            self.resamp = None

        # Audio processing transforms (same as used in evaluation)
        self.crop_or_pad = at.CenterCropOrPad(40000)
        self.db_spl = 60
        self.set_dbSPL = at.DBSPLNormalizeForegroundAndBackground(self.db_spl)

        # cache speaker/tone combos that have at least two base syllables
        self.valid_pair_keys = [
            (spk, tone)
            for (spk, tone), df in self.metadata.groupby(["speaker", "tone"])
            if df.base_syllable.nunique() >= 2
        ]
        
        # Pre-generate balanced triplets if n_examples is specified and pair_mode is True
        self.triplet_list = []
        if self.pair_mode and self.n_examples is not None:
            self._generate_balanced_triplets()

    def _load_and_resample(self, file_name):
        """Load and resample audio file."""
        audio, _ = load_audio(file_name, self.tone_perfect_dir)
        audio = audio[0]  # mono
        if self.resamp is not None:
            audio = self.resamp(audio)
        return audio
    
    def _prepare_single_wav(self, wav: torch.Tensor):
        """Apply crop/pad + dB SPL normalization."""
        wav = self.crop_or_pad(wav.squeeze())
        wav, _ = self.set_dbSPL(wav, None)
        return wav
    
    def _sample_triplet_for_tone(self, tone, rng, max_attempts=1000):
        """Sample a valid triplet for a specific tone."""
        valid_for_tone = [(spk, t) for spk, t in self.valid_pair_keys if t == tone]
        
        for _ in range(max_attempts):
            # Pick a random (speaker, tone) pair
            spk, t = valid_for_tone[rng.randint(0, len(valid_for_tone))]
            
            # Get examples for this speaker/tone
            examples = self.metadata[(self.metadata.speaker == spk) & (self.metadata.tone == t)]
            base_choices = examples.base_syllable.unique()
            
            if len(base_choices) < 2:
                continue
            
            # Sample two different base syllables
            chosen_bases = rng.choice(base_choices, size=2, replace=False)
            pair_syllables = []
            
            for base in chosen_bases:
                base_examples = examples[examples.base_syllable == base]
                if len(base_examples) == 0:
                    break
                row = base_examples.sample(random_state=rng).iloc[0]
                pair_syllables.append(row.syllable)
            else:
                # Check if we can get a negative
                speaker_tones = self.metadata[self.metadata.speaker == spk].tone.unique()
                other_tones = speaker_tones[speaker_tones != tone]
                
                if len(other_tones) > 0:
                    neg_tone = rng.choice(other_tones)
                    neg_candidates = self.metadata[
                        (self.metadata.speaker == spk) & (self.metadata.tone == neg_tone)
                    ]
                    if len(neg_candidates) > 0:
                        neg_row = neg_candidates.sample(random_state=rng).iloc[0]
                        
                        return {
                            "speaker": spk,
                            "tone": tone,
                            "syllable_a": pair_syllables[0],
                            "syllable_b": pair_syllables[1],
                            "syllable_a_file": examples[examples.syllable == pair_syllables[0]].iloc[0].file_name,
                            "syllable_b_file": examples[examples.syllable == pair_syllables[1]].iloc[0].file_name,
                            "negative_meta": neg_row,
                        }
        
        return None
    
    def _generate_balanced_triplets(self):
        """Pre-generate a balanced list of triplets across all tones."""
        if not self.include_negative:
            raise ValueError("include_negative must be True for balanced triplet generation")
        
        n_per_tone = self.n_examples // len(self.tones)
        rng = np.random.RandomState(self.random_seed)
        
        print(f"Generating {n_per_tone} triplets per tone (total: {n_per_tone * len(self.tones)})...")
        
        for tone in self.tones:
            tone_triplets = []
            attempts = 0
            max_attempts = n_per_tone * 50
            
            while len(tone_triplets) < n_per_tone and attempts < max_attempts:
                attempts += 1
                triplet = self._sample_triplet_for_tone(tone, rng)
                
                if triplet is not None:
                    # Check for duplicates (same speaker, syllables, negative)
                    is_duplicate = any(
                        t["speaker"] == triplet["speaker"] and
                        t["syllable_a"] == triplet["syllable_a"] and
                        t["syllable_b"] == triplet["syllable_b"] and
                        t["negative_meta"].tone == triplet["negative_meta"].tone
                        for t in tone_triplets
                    )
                    
                    if not is_duplicate:
                        tone_triplets.append(triplet)
            
            print(f"  Tone {tone}: generated {len(tone_triplets)}/{n_per_tone} triplets")
            self.triplet_list.extend(tone_triplets)
        
        # Shuffle the final list for randomness, but keep it reproducible
        rng_shuffle = np.random.RandomState(self.random_seed + 999)
        rng_shuffle.shuffle(self.triplet_list)
        
        print(f"Total triplets generated: {len(self.triplet_list)}")

    def __len__(self):
        if self.pair_mode and self.n_examples is not None:
            return len(self.triplet_list)
        return len(self.metadata)

    def __getitem__(self, idx):
        # If we have pre-generated triplets, use them
        if self.pair_mode and self.n_examples is not None and len(self.triplet_list) > 0:
            if idx >= len(self.triplet_list):
                raise IndexError(f"Index {idx} out of range for {len(self.triplet_list)} triplets")
            
            triplet_spec = self.triplet_list[idx]
            
            # Load and process audio files
            anchor = self._prepare_single_wav(self._load_and_resample(triplet_spec["syllable_a_file"]))
            positive = self._prepare_single_wav(self._load_and_resample(triplet_spec["syllable_b_file"]))
            negative = self._prepare_single_wav(self._load_and_resample(triplet_spec["negative_meta"].file_name))
            
            # Return in format matching NSynth: (clips dict, sr, triplet dict)
            clips = {
                "anchor": anchor,
                "positive": positive,
                "negative": negative,
            }
            
            # Triplet metadata in nested format matching NSynth structure
            triplet = {
                "anchor": {
                    "tone": triplet_spec["tone"],
                    "speaker": triplet_spec["speaker"],
                    "syllable": triplet_spec["syllable_a"],
                    "syllable_file": triplet_spec["syllable_a_file"],
                },
                "positive": {
                    "tone": triplet_spec["tone"],
                    "speaker": triplet_spec["speaker"],
                    "syllable": triplet_spec["syllable_b"],
                    "syllable_file": triplet_spec["syllable_b_file"],
                },
                "negative": {
                    "tone": triplet_spec["negative_meta"].tone,
                    "speaker": triplet_spec["speaker"],
                    "syllable": triplet_spec["negative_meta"].syllable,
                    "syllable_file": triplet_spec["negative_meta"].file_name,
                },
            }
            
            return clips, self.resample_sr, triplet
        
        # Original random sampling behavior (for backward compatibility)
        rng = np.random.RandomState(idx)

        if self.pair_mode:
            if len(self.valid_pair_keys) == 0:
                raise RuntimeError(
                    "No speaker/tone combinations with >=2 base syllables available."
                )

            # pick a talker/tone with at least two base syllables
            talker_id, tone = self.valid_pair_keys[
                rng.randint(0, len(self.valid_pair_keys))
            ]
            examples = self.metadata[
                (self.metadata.speaker == talker_id) & (self.metadata.tone == tone)
            ]
            base_choices = examples.base_syllable.unique()
            if len(base_choices) < 2:
                raise RuntimeError(
                    f"Insufficient base syllable variety for speaker {talker_id}, tone {tone}."
                )

            # robustly sample two bases that actually have rows
            max_tries = 10
            for _ in range(max_tries):
                chosen_bases = rng.choice(base_choices, size=2, replace=False)
                pair_audio, pair_syllables = [], []
                ok = True
                for base in chosen_bases:
                    base_examples = examples[examples.base_syllable == base]
                    if len(base_examples) == 0:
                        ok = False
                        break
                    row = base_examples.sample(random_state=rng).iloc[0]
                    pair_audio.append(self._load_and_resample(row.file_name))
                    pair_syllables.append(row.syllable)
                if ok:
                    break
            else:
                raise RuntimeError("Failed to sample two valid base syllables for this speaker/tone")

            negative_audio = None
            negative_meta = None
            if self.include_negative:
                neg_candidates = self.metadata[
                    (self.metadata.speaker == talker_id) & (self.metadata.tone != tone)
                ]
                if len(neg_candidates) > 0:
                    neg_row = neg_candidates.sample(random_state=rng).iloc[0]
                    negative_audio = self._load_and_resample(neg_row.file_name)
                    negative_meta = neg_row

            return {
                "audio_pair": pair_audio,
                "pair_syllables": pair_syllables,
                "pair_tone": tone,
                "speaker": talker_id,
                "negative_audio": negative_audio,
                "negative_meta": negative_meta,
            }

        # Original triplet sampling for tones 2/3/4 from same base syllable
        talker_id = rng.choice(self.talker_ids)
        base_syllable = rng.choice(self.base_syllables)
        examples = self.metadata[
            (self.metadata.speaker == talker_id)
            & (self.metadata.base_syllable == base_syllable)
        ]
        examples = examples.sort_values(by="tone")
        tone_audio = []
        syllables = examples.syllable.values
        for ix in range(min(3, len(examples))):
            tone_audio.append(self._load_and_resample(examples.iloc[ix].file_name))

        return {"audio": tone_audio, "syllables": syllables}
