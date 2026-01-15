import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import torchaudio


class NsynthTripletDataset(torch.utils.data.Dataset):
    """Self-contained NSynth interval-triplet generator with resampling."""

    def __init__(
        self,
        nsynth_root: Path = Path("/mnt/home/igriffith/ceph/datasets/nsynth"),
        split: str = "valid",
        target_sr: int = 20_000,
        n_examples: int = 300,
        seed: int = 0,
        note_duration: float = 0.5,
        lead_in: float = 0.25,
        mid_gap: float = 0.4,
        total_clip: float = 2.0,
        fade_ms: float = 50.0,
        min_interval: int = 1,
        max_interval: int = 12,
        min_midi: int = 40,
        max_midi: int = 72,
        experiment_type: str = "interval_match",
    ):
        super().__init__()
        self.nsynth_root = Path(nsynth_root)
        self.split = split
        self.target_sr = target_sr
        self.n_examples = n_examples
        self.seed = seed
        self.note_duration = note_duration
        self.lead_in = lead_in
        self.mid_gap = mid_gap
        self.total_clip = total_clip
        self.fade_ms = fade_ms
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.min_midi = min_midi
        self.max_midi = max_midi
        self.experiment_type = experiment_type

        if experiment_type not in ["interval_match", "direction_match", "instrument_match"]:
            raise ValueError(f"experiment_type must be 'interval_match' or 'direction_match' or 'instrument_match', got {experiment_type}")

        self.metadata_path = self.nsynth_root / f"nsynth-{split}" / "examples.json"
        self.audio_dir = self.nsynth_root / f"nsynth-{split}" / "audio"
        assert self.metadata_path.exists(), f"Missing metadata: {self.metadata_path}"
        assert self.audio_dir.exists(), f"Missing audio dir: {self.audio_dir}"

        with self.metadata_path.open() as f:
            self.metadata: Dict[str, Dict] = json.load(f)

        self.pitch_instrument_to_ids: Dict[int, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.instruments = set()
        for file_id, meta in self.metadata.items():
            pitch = int(meta["pitch"])
            instrument = meta["instrument_str"]
            self.pitch_instrument_to_ids[pitch][instrument].append(file_id)
            self.instruments.add(instrument)
        self.instruments = sorted(self.instruments)
        self.available_pitches = sorted(self.pitch_instrument_to_ids)

        pos = list(range(self.min_interval, self.max_interval + 1))
        neg = [-i for i in pos]
        self.interval_choices = pos + neg

    # ---------- helpers ----------
    def _load_audio(self, file_id: str) -> Tuple[torch.Tensor, int]:
        audio_path = self.audio_dir / f"{file_id}.wav"
        waveform, sr = torchaudio.load(str(audio_path))
        if waveform.shape[0] > 1:
            waveform = waveform[0]
        return waveform.squeeze(0), sr

    def _crop_with_cosine_fade(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        target_samples = int(sr * self.note_duration)
        if audio.numel() >= target_samples:
            audio = audio[:target_samples]
        else:
            audio = F.pad(audio, (0, target_samples - audio.numel()))
        ramp_samples = int(sr * self.fade_ms / 1000.0)
        if ramp_samples > 0 and target_samples >= ramp_samples:
            t = torch.linspace(0, math.pi, ramp_samples, device=audio.device)
            fade = 0.5 * (1 + torch.cos(t))
            audio[-ramp_samples:] = audio[-ramp_samples:] * fade
        return audio

    def _valid_interval_starts(self, interval: int, instrument: str) -> List[int]:
        starts: List[int] = []
        for p in self.available_pitches:
            if not (self.min_midi <= p <= self.max_midi):
                continue
            target = p + interval
            if not (self.min_midi <= target <= self.max_midi):
                continue
            if instrument in self.pitch_instrument_to_ids.get(p, {}) and instrument in self.pitch_instrument_to_ids.get(target, {}):
                starts.append(p)
        return starts

    def _choose_file(self, pitch: int, instrument: str, rng: random.Random) -> str:
        return rng.choice(self.pitch_instrument_to_ids[pitch][instrument])

    def _build_triplet(self, interval: int, rng: random.Random) -> Dict[str, Dict]:
        if interval == 0 or abs(interval) < self.min_interval or abs(interval) > self.max_interval:
            raise ValueError(f"Interval must be in ±[{self.min_interval},{self.max_interval}] and non-zero")

        if self.experiment_type == "interval_match":
            return self._build_interval_match_triplet(interval, rng)
        elif self.experiment_type == "direction_match":
            return self._build_direction_match_triplet(interval, rng)
        elif self.experiment_type == "instrument_match":
            return self._build_instrument_match_triplet(interval, rng)
        else:
            raise ValueError(f"Unknown experiment_type: {self.experiment_type}")

    def _build_interval_match_triplet(self, interval: int, rng: random.Random) -> Dict[str, Dict]:
        """Original behavior: anchor and positive share same interval and instrument, different start notes."""
        instrument_to_starts: Dict[str, List[int]] = {}
        for inst in self.instruments:
            starts = self._valid_interval_starts(interval, inst)
            if len(starts) >= 2:
                instrument_to_starts[inst] = starts
        if not instrument_to_starts:
            raise RuntimeError(f"No instruments support interval {interval}")

        instrument = rng.choice(list(instrument_to_starts.keys()))
        starts = instrument_to_starts[instrument]
        anchor_start = rng.choice(starts)
        positive_start = rng.choice([p for p in starts if p != anchor_start])
        anchor_target = anchor_start + interval
        positive_target = positive_start + interval

        candidate_neg = []
        for delta in range(-self.max_interval, self.max_interval + 1):
            if delta == 0 or delta == interval:
                continue
            t_pitch = positive_start + delta
            if self.min_midi <= t_pitch <= self.max_midi and instrument in self.pitch_instrument_to_ids.get(t_pitch, {}):
                candidate_neg.append(delta)
        if not candidate_neg:
            raise RuntimeError("No negative interval available")
        negative_interval = rng.choice(candidate_neg)
        negative_target = positive_start + negative_interval

        triplet = {
            "anchor": {
                "instrument": instrument,
                "start_pitch": anchor_start,
                "target_pitch": anchor_target,
                "interval": interval,
                "start_id": self._choose_file(anchor_start, instrument, rng),
                "target_id": self._choose_file(anchor_target, instrument, rng),
            },
            "positive": {
                "instrument": instrument,
                "start_pitch": positive_start,
                "target_pitch": positive_target,
                "interval": interval,
                "start_id": self._choose_file(positive_start, instrument, rng),
                "target_id": self._choose_file(positive_target, instrument, rng),
            },
            "negative": {
                "instrument": instrument,
                "start_pitch": positive_start,
                "target_pitch": negative_target,
                "interval": negative_interval,
                "start_id": None,
                "target_id": self._choose_file(negative_target, instrument, rng),
            },
        }
        triplet["negative"]["start_id"] = triplet["positive"]["start_id"]
        return triplet

    def _build_direction_match_triplet(self, interval: int, rng: random.Random) -> Dict[str, Dict]:
        """New behavior: anchor and positive share same MIDI values but different instruments.
        Negative uses same instrument as positive but swaps the order of MIDI notes."""
        # Find instruments that support the interval
        instrument_to_starts: Dict[str, List[int]] = {}
        for inst in self.instruments:
            starts = self._valid_interval_starts(interval, inst)
            if len(starts) >= 1:
                instrument_to_starts[inst] = starts
        if len(instrument_to_starts) < 2:
            raise RuntimeError(f"Need at least 2 instruments supporting interval {interval}, found {len(instrument_to_starts)}")

        # Choose anchor instrument and start pitch
        anchor_instrument = rng.choice(list(instrument_to_starts.keys()))
        anchor_starts = instrument_to_starts[anchor_instrument]
        anchor_start = rng.choice(anchor_starts)
        anchor_target = anchor_start + interval

        # Find a different instrument that has both anchor_start and anchor_target pitches
        candidate_positive_instruments = []
        for inst in self.instruments:
            if inst == anchor_instrument:
                continue
            if (inst in self.pitch_instrument_to_ids.get(anchor_start, {}) and 
                inst in self.pitch_instrument_to_ids.get(anchor_target, {})):
                candidate_positive_instruments.append(inst)
        
        if not candidate_positive_instruments:
            raise RuntimeError(f"No other instrument has both pitches {anchor_start} and {anchor_target}")

        positive_instrument = rng.choice(candidate_positive_instruments)
        
        # Positive: same MIDI values as anchor, different instrument
        positive_start = anchor_start
        positive_target = anchor_target

        # Negative: same instrument as positive, but swapped order (target → start)
        negative_start = anchor_target  # This was the target in anchor/positive
        negative_target = anchor_start  # This was the start in anchor/positive
        negative_interval = -interval  # Swapped direction

        triplet = {
            "anchor": {
                "instrument": anchor_instrument,
                "start_pitch": anchor_start,
                "target_pitch": anchor_target,
                "interval": interval,
                "start_id": self._choose_file(anchor_start, anchor_instrument, rng),
                "target_id": self._choose_file(anchor_target, anchor_instrument, rng),
            },
            "positive": {
                "instrument": positive_instrument,
                "start_pitch": positive_start,
                "target_pitch": positive_target,
                "interval": interval,
                "start_id": self._choose_file(positive_start, positive_instrument, rng),
                "target_id": self._choose_file(positive_target, positive_instrument, rng),
            },
            "negative": {
                "instrument": positive_instrument,
                "start_pitch": negative_start,
                "target_pitch": negative_target,
                "interval": negative_interval,
                "start_id": self._choose_file(negative_start, positive_instrument, rng),
                "target_id": self._choose_file(negative_target, positive_instrument, rng),
            },
        }
        return triplet

    def _build_instrument_match_triplet(self, interval: int, rng: random.Random) -> Dict[str, Dict]:
        """Instrument classification: anchor and positive share same instrument but different MIDI notes.
        Negative uses different instrument but same MIDI note as positive."""
        # Find instruments that support the interval with at least 2 different start pitches
        instrument_to_starts: Dict[str, List[int]] = {}
        for inst in self.instruments:
            starts = self._valid_interval_starts(interval, inst)
            if len(starts) >= 2:
                instrument_to_starts[inst] = starts
        
        if not instrument_to_starts:
            raise RuntimeError(f"No instruments support interval {interval} with at least 2 different start pitches")
        
        # Choose an instrument for anchor and positive
        instrument = rng.choice(list(instrument_to_starts.keys()))
        available_starts = instrument_to_starts[instrument]
        
        # Choose two different start MIDI notes for anchor and positive (same instrument)
        anchor_pitch = rng.choice(available_starts)
        positive_pitch = rng.choice([p for p in available_starts if p != anchor_pitch])
        
        # For the triplet structure, we need start and target pitches
        # We'll use the interval to determine target pitches
        anchor_start = anchor_pitch
        anchor_target = anchor_start + interval
        positive_start = positive_pitch
        positive_target = positive_start + interval
        negative_start = positive_pitch  # Same MIDI note as positive
        negative_target = negative_start + interval
        
        # Verify anchor and positive pitches are valid
        if not (self.min_midi <= anchor_target <= self.max_midi):
            raise RuntimeError(f"Anchor target pitch {anchor_target} out of range")
        if not (self.min_midi <= positive_target <= self.max_midi):
            raise RuntimeError(f"Positive target pitch {positive_target} out of range")
        if not (self.min_midi <= negative_target <= self.max_midi):
            raise RuntimeError(f"Negative target pitch {negative_target} out of range")
        
        # Verify anchor and positive pitches exist for their instrument
        if instrument not in self.pitch_instrument_to_ids.get(anchor_target, {}):
            raise RuntimeError(f"Instrument {instrument} doesn't have pitch {anchor_target}")
        if instrument not in self.pitch_instrument_to_ids.get(positive_target, {}):
            raise RuntimeError(f"Instrument {instrument} doesn't have pitch {positive_target}")
        
        # Choose a different instrument that has both the same start MIDI note as positive
        # and the corresponding target pitch
        candidate_negative_instruments = []
        for inst in self.instruments:
            if inst == instrument:
                continue
            if (inst in self.pitch_instrument_to_ids.get(negative_start, {}) and 
                inst in self.pitch_instrument_to_ids.get(negative_target, {})):
                candidate_negative_instruments.append(inst)
        
        if not candidate_negative_instruments:
            raise RuntimeError(f"No other instrument has both pitches {negative_start} and {negative_target}")
        
        negative_instrument = rng.choice(candidate_negative_instruments)
        
        triplet = {
            "anchor": {
                "instrument": instrument,
                "start_pitch": anchor_start,
                "target_pitch": anchor_target,
                "interval": interval,
                "start_id": self._choose_file(anchor_start, instrument, rng),
                "target_id": self._choose_file(anchor_target, instrument, rng),
            },
            "positive": {
                "instrument": instrument,
                "start_pitch": positive_start,
                "target_pitch": positive_target,
                "interval": interval,
                "start_id": self._choose_file(positive_start, instrument, rng),
                "target_id": self._choose_file(positive_target, instrument, rng),
            },
            "negative": {
                "instrument": negative_instrument,
                "start_pitch": negative_start,
                "target_pitch": negative_target,
                "interval": interval,
                "start_id": self._choose_file(negative_start, negative_instrument, rng),
                "target_id": self._choose_file(negative_target, negative_instrument, rng),
            },
        }
        return triplet

    def _make_interval_clip(self, start_id: str, target_id: str) -> Tuple[torch.Tensor, int]:
        start_audio, sr_start = self._load_audio(start_id)
        target_audio, sr_target = self._load_audio(target_id)
        if sr_start != sr_target:
            target_audio = torchaudio.functional.resample(target_audio.unsqueeze(0), sr_target, sr_start).squeeze(0)
        sr = sr_start
        start_audio = self._crop_with_cosine_fade(start_audio, sr)
        target_audio = self._crop_with_cosine_fade(target_audio, sr)
        lead_in = torch.zeros(int(sr * self.lead_in))
        gap = torch.zeros(int(sr * self.mid_gap))
        tail = max(0, int(sr * (self.total_clip - (self.lead_in + 2 * self.note_duration + self.mid_gap))))
        tail_pad = torch.zeros(tail)
        clip = torch.cat([lead_in, start_audio, gap, target_audio, tail_pad])
        target_total = int(sr * self.total_clip)
        if clip.numel() > target_total:
            clip = clip[:target_total]
        elif clip.numel() < target_total:
            clip = torch.cat([clip, torch.zeros(target_total - clip.numel())])
        return clip, sr

    def _load_triplet_audio(self, triplet: Dict[str, Dict]) -> Tuple[Dict[str, torch.Tensor], int]:
        clips: Dict[str, torch.Tensor] = {}
        sr_out = None
        for role in ("anchor", "positive", "negative"):
            clip, sr = self._make_interval_clip(triplet[role]["start_id"], triplet[role]["target_id"])
            if sr_out is None:
                sr_out = sr
            elif sr_out != sr:
                raise ValueError("Sample rate mismatch across clips")
            clips[role] = clip
        return clips, sr_out

    # ---------- dataset API ----------
    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):
        rng = random.Random(self.seed + idx)
        for _ in range(16):
            interval = rng.choice(self.interval_choices)
            try:
                triplet = self._build_triplet(interval, rng)
                clips, sr = self._load_triplet_audio(triplet)
                if sr != self.target_sr:
                    for k, v in clips.items():
                        clips[k] = torchaudio.functional.resample(v.unsqueeze(0), sr, self.target_sr).squeeze(0)
                    sr = self.target_sr
                return clips, sr, triplet
            except Exception:
                continue
        raise RuntimeError("Failed to sample a valid triplet after multiple attempts")

    @staticmethod
    def describe_triplet(triplet: Dict[str, Dict]) -> None:
        for role in ("anchor", "positive", "negative"):
            entry = triplet[role]
            print(
                f"{role.capitalize():<8} | instrument_str {entry['instrument']:<16} | "
                f"{entry['start_pitch']:>3} → {entry['target_pitch']:>3} "
                f"(Δ {entry['interval']:+d}) | ids {entry['start_id']} → {entry['target_id']}"
            )

