"""
NSynth Dataset for linear evaluation tasks.

Supports instrument family, pitch, and instrument classification tasks.
"""

import json
import torch
import torchaudio
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
from torchaudio.transforms import Resample

from default_paths import NSYNTH_DIR, require_path


class NsynthDataset(torch.utils.data.Dataset):
    """
    NSynth dataset for linear evaluation.
    
    Args:
        nsynth_root: Root directory containing extracted NSynth splits
        split: One of 'train', 'valid', 'test'
        task: One of 'family', 'pitch', 'instrument', or 'other'
        label_field: If task='other', specify the metadata field to use as label
        sample_rate: Target sample rate for audio (default: 20000 to match existing models)
        duration: Duration in seconds to crop/pad audio (None = use full length)
        fade_window_duration: Duration in seconds for Hann window fade-out at right edge (default: 0.01 = 10ms)
    """
    
    # NSynth instrument families (11 classes)
    INSTRUMENT_FAMILIES = [
        'bass', 'brass', 'flute', 'guitar', 'keyboard', 
        'mallet', 'organ', 'reed', 'string', 'synth_lead', 'vocal'
    ]
    
    def __init__(
        self,
        nsynth_root: str | None = None,
        split: str = "train",
        task: str = "family",
        label_field: Optional[str] = None,
        sample_rate: int = 20000,
        duration: Optional[float] = None,
        fade_window_duration: float = 0.01,  # 10ms default
    ):
        super().__init__()
        
        self.nsynth_root = Path(
            nsynth_root
            if nsynth_root is not None
            else require_path(NSYNTH_DIR, "COCHDNN_NSYNTH_DIR", "NSynth dataset")
        )
        self.split = split
        self.task = task
        self.sample_rate = sample_rate
        self.duration = duration
        self.fade_window_duration = fade_window_duration
        
        # Validate split
        if split not in ['train', 'valid', 'test']:
            raise ValueError(f"split must be one of ['train', 'valid', 'test'], got {split}")
        
        # Validate task
        if task == 'other' and label_field is None:
            raise ValueError("label_field must be specified when task='other'")
        
        # Load metadata
        examples_json = self.nsynth_root / f"nsynth-{split}" / "examples.json"
        if not examples_json.exists():
            raise FileNotFoundError(
                f"NSynth metadata not found at {examples_json}. "
                "Run data_scripts/extract_nsynth.py first."
            )
        
        with open(examples_json, 'r') as f:
            self.metadata = json.load(f)
        
        # Get audio directory
        self.audio_dir = self.nsynth_root / f"nsynth-{split}" / "audio"
        if not self.audio_dir.exists():
            raise FileNotFoundError(f"Audio directory not found at {self.audio_dir}")
        
        # Build file list and labels
        self.file_ids = sorted(self.metadata.keys())
        self.labels, self.label_to_idx, self.num_classes = self._build_labels()
        
        # Resampler (will be created on first use if needed)
        self.resampler = None
        self._original_sample_rate = None
        
    def _build_labels(self) -> Tuple[List[int], Dict, int]:
        """
        Build label mappings based on task.
        
        Returns:
            labels: List of integer labels for each sample
            label_to_idx: Mapping from label value to class index
            num_classes: Number of classes
        """
        labels = []
        unique_labels = set()
        
        for file_id in self.file_ids:
            metadata = self.metadata[file_id]
            
            if self.task == 'family':
                label_str = metadata['instrument_family_str']
            elif self.task == 'pitch':
                # Pitch is already an integer (MIDI note 0-127)
                label_str = str(metadata['pitch'])
            elif self.task == 'instrument':
                label_str = metadata['instrument_str']
            elif self.task == 'other':
                if self.label_field not in metadata:
                    raise ValueError(f"Field '{self.label_field}' not found in metadata")
                label_str = str(metadata[self.label_field])
            else:
                raise ValueError(f"Unknown task: {self.task}")
            
            unique_labels.add(label_str)
            labels.append(label_str)
        
        # Create mapping from label to index
        sorted_labels = sorted(unique_labels)
        label_to_idx = {label: idx for idx, label in enumerate(sorted_labels)}
        
        # Convert string labels to integer indices
        int_labels = [label_to_idx[label] for label in labels]
        
        num_classes = len(sorted_labels)
        
        return int_labels, label_to_idx, num_classes
    
    def _load_audio(self, file_id: str) -> torch.Tensor:
        """
        Load and preprocess audio file.
        
        Args:
            file_id: NSynth file ID (without extension)
            
        Returns:
            Audio tensor of shape (sample_rate * duration,) or (length,)
        """
        audio_path = self.audio_dir / f"{file_id}.wav"
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Load audio
        waveform, orig_sr = torchaudio.load(str(audio_path))
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if orig_sr != self.sample_rate:
            if self.resampler is None or self._original_sample_rate != orig_sr:
                self.resampler = Resample(orig_freq=orig_sr, new_freq=self.sample_rate)
                self._original_sample_rate = orig_sr
            waveform = self.resampler(waveform)
        
        # Remove channel dimension if single channel
        if waveform.shape[0] == 1:
            waveform = waveform.squeeze(0)
        
        # Crop or pad to desired duration
        if self.duration is not None:
            target_length = int(self.duration * self.sample_rate)
            current_length = waveform.shape[0]
            
            if current_length > target_length:
                # Take the first N seconds
                waveform = waveform[:target_length]
                # Apply Hann window to the right edge to avoid discontinuities
                window_samples = int(self.fade_window_duration * self.sample_rate)
                if window_samples > 0 and window_samples < target_length:
                    # Create Hann window for the right edge
                    hann_window = torch.hann_window(window_samples * 2, periodic=False)
                    # Use only the right half (fade-out part)
                    fade_out = hann_window[window_samples:]
                    # Apply fade-out to the last window_samples
                    waveform[-window_samples:] = waveform[-window_samples:] * fade_out
            elif current_length < target_length:
                # Pad with zeros
                pad_length = target_length - current_length
                waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        
        return waveform
    
    def __len__(self) -> int:
        return len(self.file_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample.
        
        Returns:
            waveform: Audio tensor
            label: Integer class label
        """
        file_id = self.file_ids[idx]
        waveform = self._load_audio(file_id)
        label = self.labels[idx]
        
        return waveform, label
    
    def get_class_names(self) -> List[str]:
        """Get sorted list of class names."""
        if self.task == 'family':
            return sorted(self.label_to_idx.keys())
        else:
            # For other tasks, return sorted label strings
            return sorted(self.label_to_idx.keys())


class NsynthDataModule:
    """
    Lightning-style DataModule for NSynth (optional wrapper).
    
    This is a simple wrapper that creates train/val/test dataloaders.
    """
    
    def __init__(
        self,
        nsynth_root: str | None = None,
        task: str = "family",
        label_field: Optional[str] = None,
        sample_rate: int = 20000,
        duration: Optional[float] = None,
        fade_window_duration: float = 0.01,
        batch_size: int = 256,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        self.nsynth_root = (
            nsynth_root
            if nsynth_root is not None
            else str(require_path(NSYNTH_DIR, "COCHDNN_NSYNTH_DIR", "NSynth dataset"))
        )
        self.task = task
        self.label_field = label_field
        self.sample_rate = sample_rate
        self.duration = duration
        self.fade_window_duration = fade_window_duration
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Create datasets
        self.train_dataset = NsynthDataset(
            nsynth_root=self.nsynth_root,
            split='train',
            task=task,
            label_field=label_field,
            sample_rate=sample_rate,
            duration=duration,
            fade_window_duration=fade_window_duration,
        )
        
        self.val_dataset = NsynthDataset(
            nsynth_root=self.nsynth_root,
            split='valid',
            task=task,
            label_field=label_field,
            sample_rate=sample_rate,
            duration=duration,
            fade_window_duration=fade_window_duration,
        )
        
        self.test_dataset = NsynthDataset(
            nsynth_root=self.nsynth_root,
            split='test',
            task=task,
            label_field=label_field,
            sample_rate=sample_rate,
            duration=duration,
            fade_window_duration=fade_window_duration,
        )
        
        # Get num_classes from train dataset
        self.num_classes = self.train_dataset.num_classes
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

