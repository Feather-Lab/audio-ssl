import torch 
import lightning as L
import yaml
import sys, os
import pickle
from lightning_classifier import LitWordAudioSetModel 
from lightning_ssl import LitAudioSSL 
from lightning_classifier_matched_speech_in_noise import LitWordAudioSetModel as LitWordAudioSetModelMatched
from jsinV3DataLoader_precombined_batched import jsinV3_precombined_all_signals
from torchmetrics.classification import BinaryPrecision
import robustness.audio_functions.audio_transforms as at
from tqdm import tqdm
from pathlib import Path 
import pathlib
import numpy as np 
from argparse import ArgumentParser

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

transforms = at.AudioCompose([
                    at.AudioToTensor(),
                    at.DBSPLNormalizeForegroundAndBackground(60),
                    at.UnsqueezeAudio(dim=0) # dim=0 here so batches of audio from dataloader will be (Batch, 1, Time)
                ])

def collate_fn(batch):
    batch = batch[0] # unbox wrapper added by dataloader 
    speech_batch = []
    noise_batch = []
    for (speech, noise) in zip(*batch[:2]):
        speech = transforms(speech, None)[0]
        noise = transforms(noise, None)[0]
        # Temp hack - use silence for "None" labeled examples
        # Will mask in metric calculation
        if speech is None:
            speech = torch.zeros(1,40_000)
        if noise is None:
            noise = torch.zeros(1,40_000)
        speech_batch.append(speech)
        noise_batch.append(noise)

    speech_batch = torch.cat(speech_batch).unsqueeze(1)
    noise_batch = torch.cat(noise_batch).unsqueeze(1)
    
    labels = batch[-1] # labels already collated 
    # convert labels to torch tensors 
    if isinstance(labels, dict):
        for task_key, task_labels in labels.items():
            labels[task_key] = torch.from_numpy(task_labels)
    else:
        labels = torch.from_numpy(labels) 
    # convert signal and noise into signal
    return speech_batch, noise_batch, labels 


def cli_main(args):
    L.seed_everything(args.random_seed)

    if args.config_path != "":
        config_path = pathlib.Path(args.config_path)
    elif args.config_list_path != "":
        with open(args.config_list_path, 'rb') as f:
            config_dict = pickle.load(f)
            config_path = pathlib.Path(config_dict[args.array_ix])

    print(f"Evaluating config: {config_path}")
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    config['num_workers'] = args.num_workers
    config['hparas']['batch_size'] = args.batch_size
    config['data']['eval_max'] = -1

    if args.ckpt_path == "":
        checkpoint_dir = Path(args.model_ckpt_dir) / f"{config_path.stem}/checkpoints"
        ckpt_paths = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getctime)
        ckpt_path = ckpt_paths[-1] # get latest checkpoint 
        print(ckpt_path)
    else:
        ckpt_path = args.ckpt_path


    #TODO: Update import logic for all modules needed 

    if 'ssl' in config_path.stem:
        module = LitAudioSSL

    elif config['data'].get('dataset', False) == "MatchedSpeechInNoiseDatasetBatched":
        module = LitWordAudioSetModelMatched 

    else:
        module = LitWordAudioSetModelMatched
    
    model = module.load_from_checkpoint(checkpoint_path=ckpt_path, config=config)
    model = model.eval().cuda()

    model_word_key = [key for key in config['data']['target_keys'] if 'word' in key][0]
    model_speaker_key = [key for key in config['data']['target_keys'] if 'speaker' in key][0]
    model_noise_key = [key for key in config['data']['target_keys'] if 'noise' in key][0]

    val_dataset = jsinV3_precombined_all_signals(root="/mnt/ceph/users/jfeather/data/training_datasets_audio/JSIN_all_v3/subsets/",
                                                 train=False,
                                                 transform=None,
                                                 batch_size=config['hparas']['batch_size'],
                                                 eval_max=-1)
    dataloader = torch.utils.data.DataLoader(
                                            val_dataset,
                                            batch_size=1,
                                            num_workers=config['num_workers'],
                                            shuffle=False,
                                            collate_fn=collate_fn)
    prec = BinaryPrecision()

    word_acc = []
    speaker_acc = []
    noise_prec = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            speech, noise, labels = batch
            ### Get word and speaker outs
            speech_logits = model(speech.cuda())

            word_preds = speech_logits[model_word_key].softmax(-1).argmax(-1).cpu()
            speaker_preds = speech_logits[model_speaker_key].softmax(-1).argmax(-1).cpu()

            word_acc.append((word_preds == labels["signal/word_int"]).numpy().mean()) 
            speaker_acc.append((speaker_preds == labels["signal/speaker_int"]).numpy().mean()) 

            ### Get noise label outs 
            noise_logits = model(noise.cuda())[model_noise_key].cpu()
            noise_prec.append(prec(noise_logits, labels['noise/labels_binary_via_int']).item())

    word_mean = np.mean(word_acc)
    speaker_mean = np.mean(speaker_acc)
    noise_mean = np.mean(noise_prec)

    output_dict = {
        'word_task_mean': word_mean,
        'speaker_task_mean': speaker_mean,
        'noise_task_mean': noise_mean,
    }

    print("Results:")
    for task_key, metric in output_dict.items():
        print(f"{task_key}: {metric:.3f}")

    # save results as .pkl 
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_filename = results_dir / f"{config_path.stem}_eval_jsiv3_all_tasks.pkl"
    print(f"Saving results to {results_filename}")
    with open(results_filename, 'wb') as handle:
        pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
          

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
        "--results_dir",
        default=pathlib.Path("./eval_jsin_results"),
        type=pathlib.Path,
        help="Directory where model results will be saved. (Default: './eval_jsin_results')",
    )
    parser.add_argument(
        "--ckpt_path",
        default='',
        type=str,
        help="Test from this checkpoint."
    )
    parser.add_argument(
        "--gpus",
        default=1,
        type=int,
        help="Number of GPUs per node to use for test. (Default: 1)",
    )
    parser.add_argument(
        "--batch_size",
        default=256,
        type=int,
        help="Batch size to use for test. (Default: 256)",
    )
    parser.add_argument(
    "--num_workers",
    default=0,
    type=int,
    help="Number of CPUs for dataloader. (Default: 0)",
    )
    parser.add_argument('--random_seed', default=0, type=int, help='Random seed')
    parser.add_argument('--array_ix', default=0, type=int, help='Slurm job array index')
    args = parser.parse_args()

    cli_main(args)
