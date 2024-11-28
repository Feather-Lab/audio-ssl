import h5py
import torch
import glob
import pickle
import numpy as np
from robustness.audio_functions import audio_transforms
import pandas as pd

# import psutil  # uncomment for tracking process in debug notebook


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

        # These TODOs are not implemented for the release. HDF5 files are
        # already shuffled, so we can run through them directly.
        # TODO: implement chunking the hdf5 file so that we can shuffle the data
        # TODO: implement shuffling the audioset and the speech separately
        # self.chunk_size = hdf5_chunk_size
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

        # print(f"start ix: {start} on pid {psutil.Process().pid}") # uncomment for notebook print statements
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

        # These TODOs are not implemented for the release. HDF5 files are
        # already shuffled, so we can run through them directly.
        # TODO: implement chunking the hdf5 file so that we can shuffle the data
        # TODO: implement shuffling the audioset and the speech separately
        # self.chunk_size = hdf5_chunk_size
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
            batch_size=1,
            transform=None,
    ):
        super().__init__()
        self.speech_files = h5py.File(speech_h5_path, 'r')
        self.noise_files = h5py.File(noise_h5_path, 'r')
        self.speech_metadata = pd.read_hdf(speech_h5_path)
        self.noise_metadata = pd.read_hdf(noise_h5_path)

        self.num_noise_files = len(self.noise_metadata)
        self.batch_size = batch_size

        self.random_crop = audio_transforms.RandomCrop(40000)
        self.matched_random_crop = audio_transforms.MatchedRandomSignalCrops(40000)
        self.matched_combiner = audio_transforms.MatchedCombineWithRandomDBSNR(low_db, high_db)
    
    def __len__(self):
        return len(self.speech_metadata) // (2 * self.batch_size)
    
    def __getitem__(self, idx):
        speech = self.speech_files['ndarray_data']['signal'][idx:idx + self.batch_size * 2]
        noise_idx = np.random.randint(self.num_noise_files - self.batch_size * 2)
        noise = self.noise_files['ndarray_data']['signal'][noise_idx:noise_idx + self.batch_size * 2]
        
        # shuffle the indices of speech and noise
        speech = speech[np.random.permutation(speech.shape[0])]
        noise = noise[np.random.permutation(noise.shape[0])]
        
        output_11, output_12, output_21, output_22 = [], [], [], []
        for i in range(self.batch_size):
            speech_1, speech_2 = speech[i * 2], speech[i * 2 + 1]
            if len(speech_1) > len(speech_2):
                speech_1, speech_2 = speech_2, speech_1
            noise_1, noise_2 = self.random_crop(noise[0]), self.random_crop(noise[1])
            noise_1, noise_2 = torch.tensor(noise_1), torch.tensor(noise_2)

            # randomly crop clips to be the same length (2 seconds = 40000 samples)
            cropped_11, cropped_21 = self.matched_random_crop(speech_1, speech_2)
            cropped_12, cropped_22 = self.matched_random_crop(speech_1, speech_2)

            cropped_11, cropped_21 = torch.tensor(cropped_11), torch.tensor(cropped_21)
            cropped_12, cropped_22 = torch.tensor(cropped_12), torch.tensor(cropped_22)

            # randomly mix the speech and noise with the same DBSNR
            combined_11, combined_21 = self.matched_combiner(cropped_11, cropped_21, noise_1, noise_1)
            combined_12, combined_22 = self.matched_combiner(cropped_12, cropped_22, noise_2, noise_2)  

            output_11.append(combined_11)
            output_12.append(combined_12)
            output_21.append(combined_21)
            output_22.append(combined_22)
        
        output_11 = torch.stack(output_11)
        output_12 = torch.stack(output_12)
        output_21 = torch.stack(output_21)
        output_22 = torch.stack(output_22)

        return output_11, output_12, output_21, output_22