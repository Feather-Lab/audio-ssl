import h5py
import torch
import glob
from . import audio_transforms
import pickle
import numpy as np
import pandas as pd


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

    def __init__(self, root, train=True, download=False, transform=None, eval_max=3):
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
            ]

        self.datasets = [
            H5Dataset(h5_file, transform, self.target_keys)
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

    def __init__(self, root, train=True, download=False, transform=None, eval_max=8):
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
            H5Dataset(h5_file, transform, self.target_keys)
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


class jsinV3_precombined_paired(torch.utils.data.ConcatDataset):
    # Makes a dataset using pre-paired speech and audioset background sounds
    # Works with hdf5 files for the jsinv3 dataset.
    hdf5_glob = "JSIN_all__run_*.h5"
    target_keys = ["signal/word_int"]

    def __init__(self, root, train=True, download=False, transform=None, eval_max=8):
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
            H5DatasetPaired(h5_file, transform, self.target_keys)
            for h5_file in self.all_hdf5_files
        ]

        super().__init__(self.datasets)
        self.rotate_index = 0

    def _rotate_splits(self):
        for dataset in self.datasets:
            dataset._rotate_splits()
        self.rotate_index += 1

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
    def __init__(self, path, transform, target_keys):
        """
        Builds a pytorch hdf5 dataset
        Args:
            path (str): location of the hdf5 dataset
        """
        self.file_path = path
        self.dataset = None
        self.transform = transform
        self.target_keys = target_keys
        # These TODOs are not implemented for the release. HDF5 files are
        # already shuffled, so we can run through them directly.
        # TODO: implement chunking the hdf5 file so that we can shuffle the data
        # TODO: implement shuffling the audioset and the speech separately
        # self.chunk_size = hdf5_chunk_size
        with h5py.File(self.file_path, "r", swmr=True) as file:
            self.dataset_len = len(file["sources"]["signal"]["signal"])

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

        # Before transforms, set the signal and the noise
        signal = self.dataset["sources"]["signal"]["signal"][index]
        noise = self.dataset["sources"]["noise"]["signal"][index]

        # Transforms will take in the signal and the noise source for this dataset
        # If no transform, just return the speech with no background
        if self.transform is not None:
            signal, noise = self.transform(signal, noise)
        if len(self.target_keys) == 1:
            target_paths = self.target_keys[0].split("/")
            target = self.dataset["sources"][target_paths[0]][target_paths[1]][index]
            if self.target_keys[0] == "noise/labels_binary_via_int":
                target = target.astype(np.float32)
        # If there are multiple keys, our target has them explicitly listed
        else:
            target = {}
            for target_key in self.target_keys:
                target_paths = target_key.split("/")
                target[target_key] = self.dataset["sources"][target_paths[0]][
                    target_paths[1]
                ][index]
                if target_key == "noise/labels_binary_via_int":
                    target[target_key] = target[target_key].astype(np.float32)

        return signal, target

    def __len__(self):
        return self.dataset_len


class H5DatasetPaired(torch.utils.data.Dataset):
    def __init__(self, path, transform, target_keys):
        """
        Builds a pytorch hdf5 dataset
        Args:
            path (str): location of the hdf5 dataset
        """
        self.file_path = path
        self.dataset = None
        self.transform = transform
        self.target_keys = target_keys
        # These TODOs are not implemented for the release. HDF5 files are
        # already shuffled, so we can run through them directly.
        # TODO: implement chunking the hdf5 file so that we can shuffle the data
        # TODO: implement shuffling the audioset and the speech separately
        # self.chunk_size = hdf5_chunk_size
        with h5py.File(self.file_path, "r", swmr=True) as file:
            self.dataset_len = len(file["sources"]["signal"]["signal"])
        if self.dataset_len % 2 == 1:
            self.dataset_len -= 1

        self.rotate_index = 0
        all_indices = list(range(self.dataset_len))
        self.split_1 = all_indices[::2]
        self.split_2 = all_indices[1::2]

    def _rotate_splits(self):
        self.split_2 = self.split_2[1:] + self.split_2[:1]
        self.rotate_index += 1

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

        # Before transforms, set the signal and the noise
        # signal_1 = self.dataset['sources']['signal']['signal'][index]
        # noise_1 = self.dataset['sources']['noise']['signal'][index]

        # signal_2 = self.dataset['sources']['signal']['signal'][(index + 1) % self.dataset_len]
        # noise_2 = self.dataset['sources']['noise']['signal'][(index + 1) % self.dataset_len]

        signal_1 = self.dataset["sources"]["signal"]["signal"][self.split_1[index]]
        noise_1 = self.dataset["sources"]["noise"]["signal"][self.split_1[index]]

        try:
            signal_2 = self.dataset["sources"]["signal"]["signal"][self.split_2[index]]
            noise_2 = self.dataset["sources"]["noise"]["signal"][self.split_2[index]]
        except IndexError:
            print(self.split_2[index])
            print(index)
            signal_2 = signal_1
            noise_2 = noise_1

        # catch if signal 2 is noise/invalid
        if signal_1.sum() == 0:
            # grab another ex
            signal_1 = self.dataset["sources"]["signal"]["signal"][
                self.split_1[index - 1]
            ]
        # catch if signal 2 is noise/invalid
        if signal_2.sum() == 0:
            # grab another ex
            signal_2 = self.dataset["sources"]["signal"]["signal"][
                self.split_2[index - 1]
            ]

        # Transforms will take in the signal and the noise source for this dataset
        # If no transform, just return the speech with no background
        if self.transform is not None:
            signal_11, noise = self.transform(signal_1, noise_1)
            signal_12, noise = self.transform(signal_1, noise_2)
            signal_21, noise = self.transform(signal_2, noise_1)
            signal_22, noise = self.transform(signal_2, noise_2)
            if signal_11 == None:
                print(f"Signal 11 is none on ix {index}")
            if signal_12 == None:
                print(f"Signal 12 is none on ix {index}")
            if signal_21 == None:
                print(f"Signal 21 is none on ix {index}")
                print(f"Signal 2 sum: {signal_2.sum()} {signal_2=}, ")
                print(f"Noise 1 sum: {noise_1.sum()} {noise_1=}")
            if signal_22 == None:
                print(f"Signal 22 is none on ix {index}")
        if len(self.target_keys) == 1:
            target_paths = self.target_keys[0].split("/")
            target_1 = self.dataset["sources"][target_paths[0]][target_paths[1]][
                self.split_1[index]
            ]
            try:
                target_2 = self.dataset["sources"][target_paths[0]][target_paths[1]][
                    self.split_2[index]
                ]
            except IndexError:
                target_2 = target_1
            if self.target_keys[0] == "noise/labels_binary_via_int":
                target_1 = target_1.astype(np.float32)
                target_2 = target_2.astype(np.float32)
        # If there are multiple keys, our target has them explicitly listed
        else:
            target_1, target_2 = {}, {}
            for target_key in self.target_keys:
                target_paths = target_key.split("/")
                target_1[target_key] = self.dataset["sources"][target_paths[0]][
                    target_paths[1]
                ][self.split_1[index]]
                try:
                    target_2[target_key] = self.dataset["sources"][target_paths[0]][
                        target_paths[1]
                    ][self.split_2[index]]
                except IndexError:
                    target_2[target_key] = target_1[target_key]
                if target_key == "noise/labels_binary_via_int":
                    target_1[target_key] = target_1[target_key].astype(np.float32)
                    target_2[target_key] = target_2[target_key].astype(np.float32)

        # for item_ in [signal_11, signal_12, signal_21, signal_22, target_1, target_2]:
        #     if item_ is None:
        #         print(f"None on index {index}")
        return signal_11, signal_12, signal_21, signal_22, target_1, target_2

    def __len__(self):
        return self.dataset_len // 2


class MatchedSpeechInNoiseDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            speech_h5_path,
            noise_h5_path,
            low_db=-10,
            high_db=10,
            transform=None,
    ):
        super().__init__()
        self.speech_files = h5py.File(speech_h5_path, 'r')
        self.noise_files = h5py.File(noise_h5_path, 'r')
        self.speech_metadata = pd.read_hdf(speech_h5_path)
        self.noise_metadata = pd.read_hdf(noise_h5_path)

        self.num_noise_files = len(self.noise_metadata)
        self.chunk_size = 2

        self.random_crop = audio_transforms.RandomCrop(40000)
        self.matched_random_crop = audio_transforms.MatchedRandomSignalCrops(40000)
        self.matched_combiner = audio_transforms.MatchedCombineWithRandomDBSNR(low_db, high_db)
    
    def __len__(self):
        return len(self.speech_metadata) // self.chunk_size
    
    def __getitem__(self, idx):
        speech = self.speech_files['ndarray_data']['signal'][idx:idx + self.chunk_size]
        noise_idx = np.random.randint(self.num_noise_files - self.chunk_size)
        noise = self.noise_files['ndarray_data']['signal'][noise_idx:noise_idx + self.chunk_size]

        speech_1, speech_2 = speech[0], speech[1]
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

        return combined_11, combined_12, combined_21, combined_22