"""
The following is an alteration of `measure_layer_activations_165_natural_sounds_pytorch.py` 
from the model_metamers_pytorch repository:
https://github.com/jenellefeather/model_metamers_pytorch/blob/f89cdad5c355081f97886863ed901fc9b34bce21/download_large_files.py#L36C1-L38C60

This script changes the original, to work with the pytorch ligthning modules used with the Audio-SSL
models in this repository (CochDNN).

Original Author: Jenelle Feather 

Below is the original readme: 
Measures the model activations for a set of sounds. Uses the layers specified in the 
build_network.py script. 

Set up to measure activations for the set of sounds used in Norman-Haignere et al. 2015. 

Please cite the assciated paper if you use these sounds. 

@article{norman2015distinct,
  title={Distinct cortical pathways for music and speech revealed by hypothesis-free voxel decomposition},
  author={Norman-Haignere, Sam and Kanwisher, Nancy G and McDermott, Josh H},
  journal={Neuron},
  volume={88},
  number={6},
  pages={1281--1296},
  year={2015},
  publisher={Elsevier}
}

"""

from __future__ import division
from scipy.io import wavfile
import os

# make sure we are using the correct plotting display. 
import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

import sys
if sys.version_info < (3,):
    from StringIO import StringIO as BytesIO
else:
    from io import BytesIO
import base64

import scipy
import pickle
import h5py
from argparse import ArgumentParser
import pathlib
import yaml

import torch
from lightning_scripts.lightning_ssl_matched_speech_in_noise import LitAudioSSL as LitAudioSSLMatched
from lightning_scripts.lightning_classifier_matched_speech_in_noise import LitWordAudioSetModel as LitWordAudioSetModelMatched
import robustness.audio_functions.audio_transforms as at
from robustness.tools.audio_helpers import load_audio_wav_resample
# from analysis_scripts.default_paths import fMRI_DATA_PATH

import itertools


# def preproc_sound_np(sound):
#     sound = sound - np.mean(sound)
#     sound = sound/np.sqrt(np.mean(sound**2))*0.1
#     sound = np.expand_dims(sound, 0)
#     sound = torch.from_numpy(sound).float().cuda()
#     return sound

transforms = at.AudioCompose([
                    at.AudioToTensor(),
                    at.DBSPLNormalizeForegroundAndBackground(60),
                    at.UnsqueezeAudio(dim=0), # dim=0 (Batch, Time)
                    at.UnsqueezeAudio(dim=0) # dim=0 (Batch, 1, Time)
                ])

fMRI_DATA_PATH = pathlib.Path("assets/fMRI_natsound_data")

def cli_main(args):
    ############Get model config############
    if args.config_path != "":
        config_path = pathlib.Path(args.config_path)
    elif args.config_list_path != "":
        with open(args.config_list_path, 'rb') as f:
            config_dict = pickle.load(f)
            config_path = pathlib.Path(config_dict[args.array_ix])

    print(f"Features from config: {config_path}")
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    ############Get model checkpoint############
    if args.ckpt_path == "":
        checkpoint_dir = pathlib.Path(args.model_ckpt_dir) / f"{config_path.stem}/checkpoints"
        ckpt_paths = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getctime)
        ckpt_path = ckpt_paths[-1] # get latest checkpoint 
    else:
        ckpt_path = args.ckpt_path
    print(f"Features from checkpoint: {ckpt_path}")

    ############LOAD NETWORK############
    if "supervised" in str(config_path):
        module = LitWordAudioSetModelMatched.load_from_checkpoint(checkpoint_path=ckpt_path, config=config)
    else:
        module = LitAudioSSLMatched.load_from_checkpoint(checkpoint_path=ckpt_path, config=config)
    all_layers = module.metamer_layers
    model = module.model.eval().cuda()

    ##############Begin Define Parameters#################
    save_features_dir = pathlib.Path(args.save_features_dir)
    model_name_modfier = f"_{args.dir_name_modifier}" if args.dir_name_modifier != '' else ''
    save_features_dir = save_features_dir / f"{config_path.stem}{model_name_modfier}"
    if not save_features_dir.is_dir():
        save_features_dir.mkdir(parents=True, exist_ok=True)
   
    #############LOAD_AUDIO################
    # contains the metatdata for the list of presented sounds (should be in the correct order)
    sound_list = np.load(os.path.join(fMRI_DATA_PATH, 'neural_stim_meta.npy'))

    wavs_location = os.path.join(fMRI_DATA_PATH, '165_natural_sounds')

    SR=20000 # Match with the networks we are building/training
    MEASURE_DUR=2
    wav_array = np.empty([165, SR*MEASURE_DUR])
    for wav_idx, wav_data in enumerate(sound_list):
        test_audio, SR = load_audio_wav_resample(os.path.join(wavs_location, wav_data[0].decode('utf-8')), DUR_SECS=MEASURE_DUR, resample_SR=SR)
        wav_array[wav_idx,:] = test_audio/np.sqrt(np.mean(test_audio**2))

    # Measure the activations for each sound for each layer, and put the input in the dictionary array. 

    filename = 'natsound_activations'
    # only use the non-fake layers
    all_layers = [e.split('_fake')[0] for e in all_layers] # Don't duplicate these since we aren't synthesizing
    new_all_layers = []
    for l_unique in all_layers:
        if l_unique not in new_all_layers:
            new_all_layers.append(l_unique)
    all_layers = new_all_layers
    net_layer_dict = {}
    net_layer_dict_full = {}
    net_h5py_file = h5py.File(os.path.join(save_features_dir, filename + '.h5'), "w")
    net_h5py_file_full = h5py.File(os.path.join(save_features_dir, filename + '_full.h5'), "w")

    # Save the list of layers to the hdf5
    net_h5py_file['layer_list'] = np.array([layer.encode("utf-8") for layer in all_layers])
    net_h5py_file_full['layer_list'] = np.array([layer.encode("utf-8") for layer in all_layers])

    for sound_idx, sound_info in enumerate(sound_list):
        ## Could probably process all sounds at once...
        sound, _ = transforms(wav_array[sound_idx,:], None)
        sound = sound.float().cuda()

        with torch.no_grad():
            predictions, rep, layer_returns = model(sound, with_latent=True) # Corresponding representation

        # Make the array have the correct size
        if sound_idx == 0:
            for layer in all_layers:
                print(layer)
                layer_shape_165 = layer_returns[layer].shape
                layer_shape_full = np.prod(np.array(layer_shape_165))
                if len(layer_shape_165)==4:
                    layer_shape_unraveled = layer_shape_165[1]*layer_shape_165[2]# don't take the time dimension into account
                else:
                    layer_shape_unraveled = layer_shape_165[1]
                net_layer_dict_full[layer] = net_h5py_file_full.create_dataset(layer, (165, layer_shape_full), dtype='float32')
                net_layer_dict[layer] = net_h5py_file.create_dataset(layer, (165, layer_shape_unraveled), dtype='float32')

        for layer_idx, layer in enumerate(all_layers):
            # time averaged features, so that they can be related to the fMRI activations
            if layer_returns[layer].ndim==4: # NCHW (W is time)
                net_layer_dict[layer][sound_idx,:] = np.mean(layer_returns[layer].cpu().detach().numpy(),3).ravel()
            else: # fully connected layers do not have a temporal component.  
                net_layer_dict[layer][sound_idx,:] = layer_returns[layer].cpu().detach().numpy().ravel()
            net_layer_dict_full[layer][sound_idx,:] = layer_returns[layer].cpu().detach().numpy().ravel()
    net_h5py_file.close()
    net_h5py_file_full.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config_path', default='', type=str, help='Path to experiment config.')
    parser.add_argument('--config_list_path', default='', type=str, help='Path to experiment config.')
    parser.add_argument(
        "--model_ckpt_dir",
        default=pathlib.Path("./model_checkpoints"),
        type=pathlib.Path,
        help="Directory where model checkpoints exists. (Default: './model_checkpoints')",
    )
    parser.add_argument(
        "--ckpt_path",
        default='',
        type=str,
        help="Test from this checkpoint."
    )
    parser.add_argument(
        "--save_features_dir",
        default=pathlib.Path("./fmri_analysis_model_features"),
        type=pathlib.Path,
        help="Directory where model features will be saved. (Default: './eval_jsin_results')",
    )
    parser.add_argument(
        "--dir_name_modifier",
        default='',
        type=str,
        help="Additional text to append to directory name (e.g. latest_ckpt)"
    )
    parser.add_argument('--array_ix', default=0, type=int, help='Slurm job array index')
    args = parser.parse_args()
    cli_main(args)
