import h5py
import torch
import glob
import pickle
import numpy as np
from robustness.audio_functions import audio_transforms
from torchaudio.transforms import Resample
import pandas as pd
import logging
import os
from default_paths import WORKING_DIRECTORY

def _resolve_h5_path(path):
    """Expand env-var placeholders used in release YAML configs."""
    resolved = os.path.expanduser(os.path.expandvars(str(path)))
    if "$" in resolved:
        raise RuntimeError(f"Unresolved environment variable in HDF5 path: {path}")
    return resolved


class jsinV3_precombined_all_signals(torch.utils.data.ConcatDataset):
    # Makes a dataset using pre-paired speech and audioset background sounds
    # Works with hdf5 files for the jsinv3 dataset. As the authors for information on
    # datafiles for training.
    hdf5_glob = "JSIN_all__run_*.h5"
    target_keys = [
        "signal/word_int",
        "signal/speaker_int",
        "noise/labels_binary_via_int",
    ]

    def __init__(
        self, root, train=True, download=False, transform=None, batch_size=1, eval_max=3, train_max=-1):
        """
        Builds the pytorch hdf5 combined dataset from the files found in the
        specified root directory.
        """
        del download

        if train:
            self.all_hdf5_files = glob.glob(root + "/train_*/" + self.hdf5_glob)
            if train_max > -1:
                self.all_hdf5_files = self.all_hdf5_files[:train_max]
        else:
            if eval_max == -1:
                self.all_hdf5_files = glob.glob(root + "/valid_*/" + self.hdf5_glob)
            else:
                self.all_hdf5_files = glob.glob(root + "/valid_*/" + self.hdf5_glob)[
                    0:eval_max
                ]

        self.datasets = [
            H5Dataset(h5_file, transform, self.target_keys, batch_size)
            for h5_file in self.all_hdf5_files
        ]

        super().__init__(self.datasets)

    def class_map(self):
        """
        Loads the mapping between the word IDX and human readable word map.
        """
        word_and_speaker_encodings = pickle.load(
            open("word_and_speaker_encodings_jsinv3.pckl", "rb")
        )
        class_map = word_and_speaker_encodings["word_idx_to_word"]
        return class_map


class jsinV3_precombined(torch.utils.data.ConcatDataset):
    # Makes a dataset using pre-paired speech and audioset background sounds
    # Works with hdf5 files for the jsinv3 dataset.
    hdf5_glob = "JSIN_all__run_*.h5"
    target_keys = ["signal/word_int"]

    def __init__(
        self, root, train=True, download=False, transform=None, batch_size=1, eval_max=8
    ):
        """
        Builds the pytorch hdf5 combined dataset from the files found in the
        specified root directory.
        """
        del download

        if train:
            self.all_hdf5_files = glob.glob(root + "/train_*/" + self.hdf5_glob)
        else:
            self.all_hdf5_files = glob.glob(root + "/valid_*/" + self.hdf5_glob)[
                0:eval_max
            ]  # Just get one set of them

        self.datasets = [
            H5Dataset(h5_file, transform, self.target_keys, batch_size)
            for h5_file in self.all_hdf5_files
        ]

        super().__init__(self.datasets)

    def class_map(self):
        """
        Loads the mapping between the word IDX and human readable word map.
        """
        word_and_speaker_encodings = pickle.load(
            open("word_and_speaker_encodings_jsinv3.pckl", "rb")
        )
        class_map = word_and_speaker_encodings["word_idx_to_word"]
        return class_map


class jsinV3_precombined_paired_batched(torch.utils.data.ConcatDataset):
    # Makes a dataset using pre-paired speech and audioset background sounds
    # Works with hdf5 files for the jsinv3 dataset.
    hdf5_glob = "JSIN_all__run_*.h5"
    target_keys = ["signal/word_int"]

    def __init__(
        self, root, train=True, download=False, transform=None, batch_size=1, eval_max=3
    ):
        """
        Builds the pytorch hdf5 combined dataset from the files found in the
        specified root directory.
        """
        del download

        if train:
            self.all_hdf5_files = glob.glob(root + "/train_*/" + self.hdf5_glob)
        else:
            if eval_max == -1:
                self.all_hdf5_files = glob.glob(root + "/valid_*/" + self.hdf5_glob)
            else:
                self.all_hdf5_files = glob.glob(root + "/valid_*/" + self.hdf5_glob)[
                    0:eval_max
                ]

        self.datasets = [
            H5DatasetPairedBatched(h5_file, transform, self.target_keys, batch_size)
            for h5_file in self.all_hdf5_files
        ]

        super().__init__(self.datasets)
        self.rotate_index = 0

    # def _rotate_splits(self):
    #     for dataset in self.datasets:
    #         dataset._rotate_splits()
    #     self.rotate_index += 1

    def class_map(self):
        """
        Loads the mapping between the word IDX and human readable word map.
        """
        word_and_speaker_encodings = pickle.load(
            open("word_and_speaker_encodings_jsinv3.pckl", "rb")
        )
        class_map = word_and_speaker_encodings["word_idx_to_word"]
        return class_map


class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, path, transform, target_keys, batch_size):
        """
        Builds a pytorch hdf5 dataset
        Args:
            path (str): location of the hdf5 dataset
        """
        self.file_path = path
        self.dataset = None
        self.transform = transform
        self.target_keys = target_keys
        self.batch_size = batch_size

        # HDF5 files are already shuffled, so the release loader streams them directly.
        with h5py.File(self.file_path, "r", swmr=True) as file:
            self.dataset_len = (
                len(file["sources"]["signal"]["signal"]) // self.batch_size
            )  # scale by batch size for dataloader

    def __getitem__(self, index):
        """
        Gets components of the hdf5 file that are used for training
        Args:
            index (int): index into the hdf5 file
        Returns:
            [signal, target] : the training audio (signal) containing the preprocessing
              which may combine the foreground and background speech, and the target idx
              specified by target_keys.
        """
        if self.dataset is None:
            self.dataset = h5py.File(
                self.file_path, "r", swmr=True
            )  # ["ndarray_data"]["signal"]
        # set up ix logic
        start = index * self.batch_size
        end = start + self.batch_size

        # Before transforms, set the signal and the noise
        signal = self.dataset["sources"]["signal"]["signal"][start:end]
        noise = self.dataset["sources"]["noise"]["signal"][start:end]

        # Transforms will take in the signal and the noise source for this dataset
        # If no transform, just return the speech with no background
        if self.transform is not None:
            signals = []
            for signal_, noise_ in zip(signal, noise):
                signal_, noise_ = self.transform(signal_, noise_)
                signals.append(signal_)
            signal = np.vstack(signals)
        if len(self.target_keys) == 1:
            target_paths = self.target_keys[0].split("/")
            target = self.dataset["sources"][target_paths[0]][target_paths[1]][
                start:end
            ]
            if self.target_keys[0] == "noise/labels_binary_via_int":
                target = target.astype(np.float32)
        # If there are multiple keys, our target has them explicitly listed
        else:
            target = {}
            for target_key in self.target_keys:
                target_paths = target_key.split("/")
                target[target_key] = self.dataset["sources"][target_paths[0]][
                    target_paths[1]
                ][start:end]
                if target_key == "noise/labels_binary_via_int":
                    target[target_key] = target[target_key].astype(np.float32)

        if self.transform is None:
            return signal, noise, target

        return signal, target

    def __len__(self):
        return self.dataset_len


class H5DatasetPairedBatched(torch.utils.data.Dataset):
    def __init__(self, path, transform, target_keys, batch_size):
        """
        Builds a pytorch hdf5 dataset. Returns signal1, signal2, noise1, noise2, label1, label2
        Args:
            path (str): location of the hdf5 dataset
        """
        self.file_path = path
        self.dataset = None
        self.transform = transform
        self.target_keys = target_keys
        self.batch_size = batch_size

        # HDF5 files are already shuffled, so the release loader streams them directly.
        with h5py.File(self.file_path, "r", swmr=True) as file:
            self.dataset_len = len(file["sources"]["signal"]["signal"])
        if self.dataset_len % 2 == 1:
            self.dataset_len -= 1

        # self.rotate_index = 0
        all_indices = list(range(self.dataset_len))
        self.split_1 = all_indices[::2]
        self.split_2 = all_indices[1::2]
        # scale dataset len after setting split indices
        self.dataset_len = (
            self.dataset_len // self.batch_size
        )  # scale by batch size for dataloader (accessed in len method)

    # def _rotate_splits(self):
    #     self.split_2 = self.split_2[1:] + self.split_2[:1]
    #     self.rotate_index += 1

    def __getitem__(self, index):
        """
        Gets components of the hdf5 file that are used for training
        Args:
            index (int): index into the hdf5 file
        Returns:
            [signal, target] : the training audio (signal) containing the preprocessing
              which may combine the foreground and background speech, and the target idx
              specified by target_keys.
        """

        # TODO: Re-write to shuffle within mini-batch
        # logic -> grab batch, shuffle, every 2 are paired
        if self.dataset is None:
            self.dataset = h5py.File(
                self.file_path, "r", swmr=True
            )  # ["ndarray_data"]["signal"]

        # set up ix logic
        start = index * self.batch_size
        end = start + self.batch_size

        # get indices from start to end for signal and noise
        signals = self.dataset["sources"]["signal"]["signal"][start:end]
        noises = self.dataset["sources"]["noise"]["signal"][start:end]

        # get random permutation of batch ixs assign signals to view 1 and 2. Will take second half of ixs as signal_2x
        permuted_batch_ixs = torch.randperm(self.batch_size)
        split_1, split_2 = torch.chunk(permuted_batch_ixs, 2)
        signal_1 = signals[split_1]
        signal_2 = signals[split_2]
        noise_1 = noises[split_1]
        noise_2 = noises[split_2]

        if len(self.target_keys) == 1:
            target_paths = self.target_keys[0].split("/")
            targets = self.dataset["sources"][target_paths[0]][target_paths[1]][
                start:end
            ]
            target_1 = targets[split_1]
            target_2 = targets[split_2]
            if self.target_keys[0] == "noise/labels_binary_via_int":
                target_1 = target_1.astype(np.float32)
                target_2 = target_2.astype(np.float32)

        # If there are multiple keys, our target has them explicitly listed
        else:
            target_1, target_2 = {}, {}
            for target_key in self.target_keys:
                target_paths = target_key.split("/")
                targets = self.dataset["sources"][target_paths[0]][target_paths[1]][
                    start:end
                ]
                target_1[target_key] = targets[split_1]
                target_2[target_key] = targets[split_2]
                if target_key == "noise/labels_binary_via_int":
                    target_1[target_key] = target_1[target_key].astype(np.float32)
                    target_2[target_key] = target_2[target_key].astype(np.float32)

        return signal_1, signal_2, noise_1, noise_2, target_1, target_2

    def __len__(self):
        return self.dataset_len // 2


class MatchedSpeechInNoiseDatasetBatched(torch.utils.data.Dataset):
    def __init__(
            self,
            speech_h5_path,
            noise_h5_path,
            low_db=-10,
            high_db=10,
            db_spl=60,
            batch_size=1,
            transform=None,
            target_keys=None,
            blocked_batches=True,
            signal_augment=False,
            skip_aug_match=False,
            clean_percentage=0.0,
            overfit=False,
    ):
        super().__init__()
        speech_h5_path = _resolve_h5_path(speech_h5_path)
        noise_h5_path = _resolve_h5_path(noise_h5_path)
        self.speech_files = h5py.File(speech_h5_path, 'r', swmr=True)
        self.noise_files = h5py.File(noise_h5_path, 'r', swmr=True)
        self.speech_metadata = pd.read_hdf(speech_h5_path)
        self.speech_metadata = self.speech_metadata.dropna() ## Removes null label 

        self.noise_metadata = pd.read_hdf(noise_h5_path)
        self.noise_metadata = self.noise_metadata.dropna() ## Removes null label 

        self.num_noise_files = len(self.noise_metadata)
        self.batch_size = batch_size
        self.target_keys = target_keys
        self.blocked_batches = blocked_batches
        # set params for clean signal sampling 
        self.clean_percentage = clean_percentage
        self.num_clean = int(batch_size * clean_percentage)

        self.signal_augment = signal_augment
        self.random_crop = audio_transforms.RandomCrop(40000)
        self.matched_random_crop = audio_transforms.MatchedRandomSignalCrops(40000, skip_aug_match=skip_aug_match)
        self.matched_combiner = audio_transforms.MatchedCombineWithRandomDBSNR(low_db, high_db)
        self.set_dbSPL = audio_transforms.DBSPLNormalizeForegroundAndBackground(db_spl)
        if signal_augment:
            self.matched_signal_augment = audio_transforms.MatchedRandomSignalAugmentSox(sample_rate=20000, skip_aug_match=skip_aug_match)

    def class_map(self):
        """
        Loads the mapping between the word IDX and human readable word map.
        """
        encoding_path = WORKING_DIRECTORY / "robustness" / "audio_functions" / "word_and_speaker_encodings_jsinv3.pckl"
        word_and_speaker_encodings = pickle.load(open(encoding_path, "rb"))
        class_map = word_and_speaker_encodings["word_idx_to_word"]
        return class_map, word_and_speaker_encodings

    def __len__(self):
        return len(self.speech_metadata) // (2 * self.batch_size)
    
    def __getitem__(self, idx):
        # Modify so batches are not sliding windows over dataset 
        if self.blocked_batches:
            start = idx * self.batch_size * 2 
            end = start + self.batch_size * 2 
        else:
            start = idx 
            end = idx + self.batch_size * 2 
        speech_ixs = np.arange(start, end)

        noise_idx = np.random.randint(self.num_noise_files - self.batch_size * 2)
        noise_ixs = np.arange(noise_idx, noise_idx + self.batch_size * 2)

        speech = self.speech_files['ndarray_data']['signal'][speech_ixs]
        noise = self.noise_files['ndarray_data']['signal'][noise_ixs]
        
        # shuffle the indices of speech and noise - externalize for label ix-ing
        speech_batch_ixs = np.random.permutation(speech.shape[0])
        noise_batch_ixs = np.random.permutation(noise.shape[0])
        
        output_11, output_12, output_21, output_22 = [], [], [], []

        pos_inf_ixs = None
        if self.clean_percentage > 0:
            pos_inf_ixs = np.random.choice(self.batch_size, size=self.num_clean, replace=False)

        if self.target_keys:
            # track labels of each combo 
            target_11 = {}
            target_12 = {}
            target_21 = {}
            target_22 = {}
            
            for target_key in self.target_keys:
                for target_set in [target_11, target_12, target_21, target_22]:
                    target_set[target_key] = []
                
        for i in range(self.batch_size):
            # map batch ix to shuffle ix to make label alignment easy
            speech_1_ix, speech_2_ix = speech_batch_ixs[i * 2], speech_batch_ixs[i * 2 + 1]
            noise_1_ix, noise_2_ix = noise_batch_ixs[i * 2] , noise_batch_ixs[i * 2 + 1]
            # get label ixs 
            speech_label_1_ix, speech_label_2_ix = speech_ixs[speech_1_ix], speech_ixs[speech_2_ix]
            noise_label_1_ix, noise_label_2_ix = noise_ixs[noise_1_ix], noise_ixs[noise_2_ix]

            # get speech example
            speech_1, speech_2 = speech[speech_1_ix], speech[speech_2_ix]
            if len(speech_1) > len(speech_2):
                speech_1, speech_2 = speech_2, speech_1
                # track ix swap for labeling 
                speech_label_1_ix, speech_label_2_ix = speech_label_2_ix, speech_label_1_ix

            # get noise example 
            noise_1, noise_2 = self.random_crop(noise[noise_1_ix]), self.random_crop(noise[noise_2_ix])
            noise_1, noise_2 = torch.tensor(noise_1), torch.tensor(noise_2)

            # store labels if supervised
            if self.target_keys:
                for target_key in self.target_keys:
                    target_type, target_name = target_key.split("/")
                    if target_type == 'signal':
                        target_11[target_key].append(self.speech_metadata.loc[speech_label_1_ix, target_name].item())
                        target_12[target_key].append(self.speech_metadata.loc[speech_label_1_ix, target_name].item())
                        target_21[target_key].append(self.speech_metadata.loc[speech_label_2_ix, target_name].item())
                        target_22[target_key].append(self.speech_metadata.loc[speech_label_2_ix, target_name].item())
                    elif target_type == 'noise':
                        target_11[target_key].append(self.noise_files['ndarray_data']['labels_binary_via_int'][noise_label_1_ix])
                        target_12[target_key].append(self.noise_files['ndarray_data']['labels_binary_via_int'][noise_label_2_ix])
                        target_21[target_key].append(self.noise_files['ndarray_data']['labels_binary_via_int'][noise_label_1_ix])
                        target_22[target_key].append(self.noise_files['ndarray_data']['labels_binary_via_int'][noise_label_2_ix])
            

            # randomly crop clips to be the same length (2 seconds = 40000 samples)
            cropped_11, cropped_21 = self.matched_random_crop(speech_1, speech_2)
            cropped_12, cropped_22 = self.matched_random_crop(speech_1, speech_2)

            # Apply pitch, tempo, and filtering augments 
            if self.signal_augment:
                cropped_11, cropped_21 = self.matched_signal_augment(cropped_11, cropped_21)
                cropped_12, cropped_22 = self.matched_signal_augment(cropped_12, cropped_22)
                # hack to kill sox warnings 
                if idx == 0:
                    logging.getLogger('sox').setLevel(logging.ERROR)

            cropped_11, cropped_21 = torch.tensor(cropped_11), torch.tensor(cropped_21)
            cropped_12, cropped_22 = torch.tensor(cropped_12), torch.tensor(cropped_22)

            # if i is in pos inf snr samples, set to None
            if self.clean_percentage > 0:
                if i in pos_inf_ixs:
                    noise_1 = None
             # randomly mix the speech and noise with the same DBSNR
            combined_11, combined_21 = self.matched_combiner(cropped_11, cropped_21, noise_1, noise_1)
            combined_12, combined_22 = self.matched_combiner(cropped_12, cropped_22, noise_2, noise_2)  

            # set dB SPL for mixtures 
            combined_11, _ = self.set_dbSPL(combined_11, None)
            combined_12, _ = self.set_dbSPL(combined_12, None)
            combined_21, _ = self.set_dbSPL(combined_21, None)
            combined_22, _ = self.set_dbSPL(combined_22, None)

            output_11.append(combined_11)
            output_12.append(combined_12)
            output_21.append(combined_21)
            output_22.append(combined_22)
        
        output_11 = torch.stack(output_11).float()
        output_12 = torch.stack(output_12).float()
        output_21 = torch.stack(output_21).float()
        output_22 = torch.stack(output_22).float()
        
        if self.target_keys:
            # format targets 
            for target in [target_11, target_12, target_21 , target_22]:
                for target_key, target_list in target.items():
                    if 'noise' in target_key:
                        target[target_key] = torch.from_numpy(np.stack(target_list, axis=0)).float()
                    else:
                        target[target_key] = torch.tensor(target_list)
        
            return [output_11, output_12, output_21, output_22], [target_11, target_12, target_21 , target_22]

        return output_11, output_12, output_21, output_22


class MatchedAudiosetBatched(torch.utils.data.Dataset):
    """
    Builds a pytorch hdf5 dataset. Returns signal1, signal2, label if train; signal1, label else
    Expects datasets in segments.hdf5 format:
        keys: ['labels', 'wav']
    Args:
        path (str): location of the hdf5 dataset
    """
    def __init__(
            self,
            noise_h5_path,
            low_db=-10,
            high_db=10,
            db_spl=60,
            batch_size=1,
            transform=None,
            target_keys=None,
            blocked_batches=True,
            signal_augment=False,
            skip_aug_match=False,
            clean_percentage=0.0,
            in_sample_rate=16_000,
            out_sample_rate=20_000,
            overfit=False,
            max_retries=10,
    ):
        super().__init__()
        noise_h5_path = _resolve_h5_path(noise_h5_path)
        self.noise_files = h5py.File(noise_h5_path, 'r', swmr=True)
        self.num_noise_files = len(self.noise_files['wav'])
        self.batch_size = batch_size
        self.target_keys = target_keys
        self.blocked_batches = blocked_batches
        self.clean_percentage = clean_percentage
        self.num_clean = int(batch_size * clean_percentage)
        self.signal_augment = signal_augment
        self.in_sample_rate = in_sample_rate
        self.out_sample_rate = out_sample_rate
        self.output_dur = int(2 * self.out_sample_rate)
        self.max_retries = max_retries
        
        # Initialize audio transforms
        self.resample_audio = Resample(orig_freq=in_sample_rate, new_freq=out_sample_rate)
        self.pad_signal = audio_transforms.PadToLen(int(5 * self.out_sample_rate))
        self.random_crop = audio_transforms.RandomCrop(self.output_dur)
        self.matched_random_crop = audio_transforms.MatchedRandomSignalCrops(
            self.output_dur, skip_aug_match=skip_aug_match
        )
        self.matched_combiner = audio_transforms.MatchedCombineWithRandomDBSNR(low_db, high_db)
        self.set_dbSPL = audio_transforms.DBSPLNormalizeForegroundAndBackground(db_spl)
        
        if signal_augment:
            self.matched_signal_augment = audio_transforms.MatchedRandomSignalAugmentSox(
                sample_rate=self.out_sample_rate, skip_aug_match=skip_aug_match
            )

    def __len__(self):
        return self.num_noise_files // (2 * self.batch_size)
    
    def _ensure_torch(self, x):
        """Convert numpy array to torch tensor if needed."""
        return torch.from_numpy(x) if isinstance(x, np.ndarray) else x
    
    def _apply_signal_augmentation(self, signal1, signal2, suppress_warnings=False):
        """Apply signal augmentation to a pair of signals."""
        if not self.signal_augment:
            return signal1, signal2
        
        if suppress_warnings:
            logging.getLogger('sox').setLevel(logging.ERROR)
        
        aug1, aug2 = self.matched_signal_augment(signal1.numpy(), signal2.numpy())
        return self._ensure_torch(aug1), self._ensure_torch(aug2)
    
    def _process_signal_pair(self, source1, source2, noise1, noise2, apply_aug=True, suppress_warnings=False):
        """Process a pair of source signals with noise."""
        # Ensure shorter signal comes first
        if len(source1) > len(source2):
            source1, source2 = source2, source1
        
        # Crop noise
        noise1 = self.random_crop(noise1)
        noise2 = self.random_crop(noise2)
        
        # Pad and crop sources
        source1, source2 = self.pad_signal(source1, source2)
        cropped1, cropped2 = self.matched_random_crop(source1, source2)
        
        # Apply augmentation
        if apply_aug:
            cropped1, cropped2 = self._apply_signal_augmentation(
                cropped1, cropped2, suppress_warnings
            )
        
        cropped1, cropped2 = self._ensure_torch(cropped1), self._ensure_torch(cropped2)
        
        # Combine with noise
        combined1, combined2 = self.matched_combiner(cropped1, cropped2, noise1, noise2)
        
        # Normalize dB SPL
        combined1, _ = self.set_dbSPL(combined1, None)
        combined2, _ = self.set_dbSPL(combined2, None)
        
        return combined1, combined2
    
    def _get_labels(self, source_label_ix, noise_label_ix, target_key):
        """Get combined labels for source and noise."""
        target_type, target_name = target_key.split("/")
        
        if target_type == 'signal':
            print(f"WARNING: labels for {target_type} not supported.")
            print("Silently continuing for now...")
            return None
        
        if target_type == 'noise':
            source_labels = self.noise_files['labels'][source_label_ix].astype('int')
            noise_labels = self.noise_files['labels'][noise_label_ix].astype('int')
            labels = torch.from_numpy(source_labels + noise_labels)
            return torch.clamp(labels, max=1)
        
        return None
    
    def _create_targets(self, source_label_1_ix, source_label_2_ix, noise_label_1_ix, noise_label_2_ix):
        """Create target dictionaries for all four combinations."""
        if not self.target_keys:
            return None
        
        # Define the four combinations
        label_combinations = [
            (source_label_1_ix, noise_label_1_ix),  # target_11
            (source_label_1_ix, noise_label_2_ix),  # target_12
            (source_label_2_ix, noise_label_1_ix),  # target_21
            (source_label_2_ix, noise_label_2_ix),  # target_22
        ]
        
        targets = [{} for _ in range(4)]
        
        for target_key in self.target_keys:
            for target_dict, (src_ix, noise_ix) in zip(targets, label_combinations):
                labels = self._get_labels(src_ix, noise_ix, target_key)
                if labels is not None:
                    target_dict[target_key] = labels
        
        return targets
    
    def _get_example(self, i, source, noise, source_batch_ixs, noise_batch_ixs, 
                     source_ixs, noise_ixs):
        """Generate one example with four combinations."""
        # Get indices
        source_1_ix, source_2_ix = source_batch_ixs[i * 2], source_batch_ixs[i * 2 + 1]
        noise_1_ix, noise_2_ix = noise_batch_ixs[i * 2], noise_batch_ixs[i * 2 + 1]
        
        # Get label indices
        source_label_1_ix = source_ixs[source_1_ix]
        source_label_2_ix = source_ixs[source_2_ix]
        noise_label_1_ix = noise_ixs[noise_1_ix]
        noise_label_2_ix = noise_ixs[noise_2_ix]
        
        # Get signals
        source_1, source_2 = source[source_1_ix], source[source_2_ix]
        noise_1, noise_2 = noise[noise_1_ix], noise[noise_2_ix]
        
        # Swap if source_1 is longer (and swap labels accordingly)
        if len(source_1) > len(source_2):
            source_1, source_2 = source_2, source_1
            source_label_1_ix, source_label_2_ix = source_label_2_ix, source_label_1_ix
        
        # Process both combinations with the same source pair
        suppress_warnings = (i == 0)  # Only suppress for first iteration
        
        combined_11, combined_21 = self._process_signal_pair(
            source_1, source_2, noise_1, noise_1, 
            apply_aug=True, suppress_warnings=suppress_warnings
        )
        combined_12, combined_22 = self._process_signal_pair(
            source_1, source_2, noise_2, noise_2, 
            apply_aug=True, suppress_warnings=False
        )
        
        # Create targets if needed
        targets = self._create_targets(
            source_label_1_ix, source_label_2_ix, 
            noise_label_1_ix, noise_label_2_ix
        )
        
        if targets:
            return combined_11, combined_12, combined_21, combined_22, targets
        return combined_11, combined_12, combined_21, combined_22
    
    def _screen_batch(self, combined_sigs):
        """Check if any signals are None and return True if resampling needed."""
        return any(sig is None for sig in combined_sigs)
    
    def _load_sources(self, idx):
        """Load and resample source audio."""
        # Determine index range
        if self.blocked_batches:
            start = idx * self.batch_size * 2
            end = start + self.batch_size * 2
        else:
            start = idx
            end = idx + self.batch_size * 2
        
        source_ixs = np.arange(start, end)
        source = self.resample_audio(torch.from_numpy(self.noise_files['wav'][source_ixs]))
        
        return source, source_ixs
    
    def __getitem__(self, idx):
        """Get a batch of examples."""
        retry_count = 0
        
        while retry_count < self.max_retries:
            # Load sources (may be reloaded on retry)
            source, source_ixs = self._load_sources(idx)
            
            # Load noise (same for all retries)
            if retry_count == 0:
                noise_idx = np.random.randint(self.num_noise_files - self.batch_size * 2)
                noise_ixs = np.arange(noise_idx, noise_idx + self.batch_size * 2)
                noise = self.resample_audio(torch.from_numpy(self.noise_files['wav'][noise_ixs]))
            
            # Shuffle indices
            source_batch_ixs = np.random.permutation(source.shape[0])
            noise_batch_ixs = np.random.permutation(noise.shape[0])
            
            # Initialize output lists
            outputs = [[] for _ in range(4)]  # output_11, output_12, output_21, output_22
            
            if self.target_keys:
                targets = [{key: [] for key in self.target_keys} for _ in range(4)]
            
            # Flag to track if we need to resample
            needs_resample = False
            
            # Generate batch
            for i in range(self.batch_size):
                batch = self._get_example(
                    i, source, noise, source_batch_ixs, noise_batch_ixs, 
                    source_ixs, noise_ixs
                )
                
                # Check if resampling is needed
                if self._screen_batch(batch[:4]):
                    needs_resample = True
                    break
                
                # Unpack batch
                if self.target_keys:
                    *combined_signals, batch_targets = batch
                else:
                    combined_signals = batch
                
                # Append signals
                for output_list, signal in zip(outputs, combined_signals):
                    output_list.append(signal)
                
                # Append targets
                if self.target_keys:
                    for target_dict, batch_target_dict in zip(targets, batch_targets):
                        for key, value in batch_target_dict.items():
                            target_dict[key].append(value)
            
            # If no resampling needed, we're done
            if not needs_resample:
                # Stack outputs
                output_tensors = [torch.stack(output_list).float() for output_list in outputs]
                
                # Stack targets if needed
                if self.target_keys:
                    for target_dict in targets:
                        for key in target_dict:
                            target_dict[key] = torch.stack(target_dict[key], dim=0).float()
                    
                    return output_tensors, targets
                
                return tuple(output_tensors)
            
            # Increment retry counter and resample sources
            retry_count += 1
            if retry_count < self.max_retries:
                # Resample source indices randomly
                idx = np.random.randint(self.__len__())
        
        # If we've exhausted retries, raise an error
        raise RuntimeError(f"Failed to generate valid batch after {self.max_retries} retries")


class CleanSpeechInNoiseValDatasetBatched(torch.utils.data.Dataset):
    def __init__(
            self,
            speech_h5_path,
            db_spl=60,
            batch_size=1,
            target_keys=None,
            return_noise=False,
            sig_len = 40_000,
            overfit=False,
    ):
        super().__init__()
        speech_h5_path = _resolve_h5_path(speech_h5_path)
        self.speech_files = h5py.File(speech_h5_path, 'r', swmr=True)
        self.speech_metadata = pd.read_hdf(speech_h5_path)
        self.speech_metadata = self.speech_metadata.dropna() ## Removes null label 
        self.return_noise = return_noise
        self.sig_len = sig_len
        self.batch_size = batch_size
        self.target_keys = target_keys
        self.center_crop = audio_transforms.CenterCropForegroundBackground(signal_size=self.sig_len, crop_length=self.sig_len)
        self.set_dbSPL = audio_transforms.DBSPLNormalizeForegroundAndBackground(db_spl)

        if self.return_noise:
            noise_h5_path = os.environ.get("COCHDNN_AUDIONOISE_H5")
            if noise_h5_path is None:
                raise RuntimeError("Set COCHDNN_AUDIONOISE_H5 to return background noise examples.")
            self.noise_files = h5py.File(noise_h5_path, 'r', swmr=True)
            self.num_noise_files = len(self.noise_files['ndarray_data']['signal'])


    def class_map(self):
        """
        Loads the mapping between the word IDX and human readable word map.
        """
        encoding_path = WORKING_DIRECTORY / "robustness" / "audio_functions" / "word_and_speaker_encodings_jsinv3.pckl"
        word_and_speaker_encodings = pickle.load(open(encoding_path, "rb"))
        class_map = word_and_speaker_encodings["word_idx_to_word"]
        return class_map, word_and_speaker_encodings

    def __len__(self):
        return len(self.speech_metadata) // (self.batch_size)
    
    def __getitem__(self, idx):
        # Modify so batches are not sliding windows over dataset 
        start = idx * self.batch_size 
        end = start + self.batch_size 

        speech_ixs = np.arange(start, end)

        speech = self.speech_files['ndarray_data']['signal'][speech_ixs]
        
        audio = []
        for speech_eg in speech:
            speech_eg, _ = self.center_crop(speech_eg, None)
            speech_eg, _ = self.set_dbSPL(torch.tensor(speech_eg), None)
            audio.append(speech_eg)
        audio = torch.stack(audio).float()
        # shuffle the indices of speech and noise - externalize for label ix-ing
        
        # track labels of each combo 
        targets = {}            
        for target_key in self.target_keys:
            targets[target_key] = []
            target_type, target_name = target_key.split("/")
            targets[target_key] = self.speech_metadata.loc[speech_ixs, target_name].values
                    # elif target_type == 'noise':
                    #     targets[target_key]  = self.noise_files['labels'][noise_label_1_ix

        if self.return_noise:
            # Sample random noise indices (same number as batch size)
            if self.num_noise_files >= self.batch_size:
                noise_idx = np.random.randint(0, self.num_noise_files - self.batch_size + 1)
                noise_ixs = np.arange(noise_idx, noise_idx + self.batch_size)
            else:
                # If noise file is smaller than batch_size, sample with replacement
                noise_ixs = np.random.randint(0, self.num_noise_files, size=self.batch_size)
            noise = self.noise_files['ndarray_data']['signal'][noise_ixs]
            noise_audio = []
            for noise_eg in noise:
                noise_eg, _ = self.center_crop(noise_eg, None)
                noise_eg = audio_transforms.pad_or_trim_to_len(noise_eg, self.sig_len, mode='both')
                noise_eg, _ = self.set_dbSPL(torch.tensor(noise_eg), None)
                noise_audio.append(noise_eg)   
            noise_audio = torch.stack(noise_audio).float()
            return audio, noise_audio, targets
        return audio, targets


class jsinv3_audioset_SSL(torch.utils.data.ConcatDataset):
    # Makes a dataset using pre-paired speech and audioset background sounds
    # Works with hdf5 files for the jsinv3 dataset.
    hdf5_glob = "JSIN_all__run_*.h5"
    target_keys = ["noise/labels_binary_via_int"]

    def __init__(
        self, root, train=True, download=False, transform=None, batch_size=1, eval_max=3
    ):
        """
        Builds the pytorch hdf5 combined dataset from the files found in the
        specified root directory.
        """
        del download

        if train:
            self.all_hdf5_files = glob.glob(root + "/train_*/" + self.hdf5_glob)
        else:
            if eval_max == -1:
                self.all_hdf5_files = glob.glob(root + "/valid_*/" + self.hdf5_glob)
            else:
                self.all_hdf5_files = glob.glob(root + "/valid_*/" + self.hdf5_glob)[
                    0:eval_max
                ]

        self.datasets = [
            H5DatasetAudiosetSSL(h5_file, transform, self.target_keys, batch_size, train=train)
            for h5_file in self.all_hdf5_files
        ]

        super().__init__(self.datasets)
        self.rotate_index = 0


    def class_map(self):
        """
        Loads the mapping between the word IDX and human readable word map.
        """
        word_and_speaker_encodings = pickle.load(
            open("word_and_speaker_encodings_jsinv3.pckl", "rb")
        )
        class_map = word_and_speaker_encodings["word_idx_to_word"]
        return class_map



class H5DatasetAudiosetSSL(torch.utils.data.Dataset):
    def __init__(self, path, transform, target_keys, batch_size, train=True):
        """
        Builds a pytorch hdf5 dataset. Returns signal1, signal2, label if train; signal1, label else
        Args:
            path (str): location of the hdf5 dataset
        """
        self.file_path = path
        self.dataset = None
        self.transform = transform
        self.target_keys = target_keys
        self.batch_size = batch_size
        self.train = train

        # These TODOs are not implemented for the release. HDF5 files are
        # already shuffled, so we can run through them directly.
        # TODO: implement chunking the hdf5 file so that we can shuffle the data
        # TODO: implement shuffling the audioset and the speech separately
        # self.chunk_size = hdf5_chunk_size
        with h5py.File(self.file_path, "r", swmr=True) as file:
            self.n_signals = len(file["sources"]["signal"]["signal"])

        # scale dataset len after setting split indices
        self.dataset_len =  self.n_signals // self.batch_size
          # scale by batch size for dataloader (accessed in len method)

    def __getitem__(self, index):
        """
        """
        if self.dataset is None:
            self.dataset = h5py.File(
                self.file_path, "r", swmr=True
            )  # ["ndarray_data"]["signal"]
        # set up ix logic
        start = index * self.batch_size
        end = start + self.batch_size # second half will be noises used for augmentation

        batch_ixs = np.arange(start, end)
        noises = self.dataset["sources"]["noise"]["signal"][batch_ixs]

        if self.train:
            aug_ix = np.random.randint(self.n_signals - self.batch_size) # second half will be noises used for augmentation
            augments = self.dataset["sources"]["noise"]["signal"][aug_ix: aug_ix + self.batch_size ]
        ## Get labels 
        target_paths = self.target_keys[0].split("/")
        target = self.dataset["sources"][target_paths[0]][target_paths[1]][
                batch_ixs
            ]

        # get where noises are None 
        bad_ixs = np.argwhere(noises.sum(1) == 0).flatten()
        if len(bad_ixs) > 0:
            good_ixs =  np.argwhere(noises.sum(1) != 0).flatten()
            samp_ixs = np.random.choice(good_ixs, size=len(bad_ixs))
            assert len(bad_ixs) == len(samp_ixs), f"{len(bad_ixs)} bad ixs, but drew {len(samp_ixs)} ixs to resample."
            noises[bad_ixs] = noises[samp_ixs]
            target[bad_ixs] = target[samp_ixs]
        
        target = torch.from_numpy(target).float()
        
        if self.train:
            aud_1 = []
            aud_2 = []
            for noise in noises:
                aug_ixs = np.random.randint(self.batch_size, size=2)
                aug_1, aug_2 = augments[aug_ixs]
                # get view
                view1, _ = self.transform(noise, aug_1)
                view2, _ = self.transform(noise, aug_2)
                aud_1.append(view1)
                aud_2.append(view2)
            aud_1 = torch.stack(aud_1)
            aud_2 = torch.stack(aud_2)

            return aud_1, aud_2, target 
        else:
            aud_1 = torch.stack([
                self.transform(noise, None)[0] for noise in noises
            ])
            return aud_1, target 
        
    def __len__(self):
        return self.dataset_len