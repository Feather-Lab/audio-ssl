
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import logging 
import robustness.audio_functions.audio_transforms as at
from lightning_scripts.jsinV3DataLoader_precombined_batched import jsinV3_precombined_all_signals
import yaml 
from lightning_scripts.lightning_ssl_matched_speech_in_noise import LitAudioSSL
from pathlib import Path 
from argparse import ArgumentParser, BooleanOptionalAction


torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class AudioAugmentStackWithParams(object):
    def __init__(self, low_db=-10, high_db=10):
        super().__init__()
        self.np_set_dbSPL = at.DBSPLNormalizeForegroundAndBackground(dbspl=60, use_np=True)
        self.torch_set_dbSPL = at.DBSPLNormalizeForegroundAndBackground(dbspl=60, use_np=False)
        self.crop = at.CenterCrop(crop_length=40_000) # crop to middle 2 seconds

        self.combine_db_snr = at.CombineWithRandomDBSNRWithParam(low_db, high_db)
        self.Pitch = at.ApplySingleAugmentSox('pitch', return_params=True)
        self.Tempo = at.ApplySingleAugmentSox('tempo', return_params=True)
        self.Filter = at.ApplySingleAugmentSox('filter', return_params=True)

    def __call__(self, aud, background):
        logging.getLogger('sox').setLevel(logging.ERROR)

        assert aud is not None, " aud is None on input"
        clean_aud = self.crop(aud)
        assert clean_aud is not None, "clean aud is None post crop"
        clean_aud, background = self.np_set_dbSPL(clean_aud, background)
        # apply augmentations
        # sox aug first to match ssl training 
        assert clean_aud is not None, "clean aud is None post set level"
        aug_aud, n_semitones =  self.Pitch(clean_aud)
        aug_aud, temp_shift = self.Tempo(aug_aud)
        aug_aud, (order, low_cutoff, high_cutoff) = self.Filter(aug_aud)
        aug_aud = torch.from_numpy(aug_aud)
        background = torch.from_numpy(background)
        aug_aud, db_snr = self.combine_db_snr(aug_aud, background)
        aug_aud, _ = self.torch_set_dbSPL(aug_aud, None)
        aug_aud = aug_aud.reshape(1,-1)
        params = torch.tensor([db_snr, n_semitones, temp_shift, order, low_cutoff, high_cutoff])
        clean_aud = torch.from_numpy(clean_aud).reshape(1,-1)
        return (clean_aud, aug_aud), params


transforms = AudioAugmentStackWithParams()

def collate_fn(batch):
    batch = batch[0] # unbox wrapper added by dataloader 
    clean_signals = []
    augmented_signals = []
    params = []
    labels = batch[-1] # labels already collated 
    # convert labels to torch tensors 
    if isinstance(labels, dict):
        for task_key, task_labels in labels.items():
            labels[task_key] = torch.from_numpy(task_labels)
    else:
        labels = torch.from_numpy(labels) 
    # convert signal and noise into signal
    for (signal, noise) in  zip(*batch[:2]):
        if signal.sum() == 0 or signal is None:
            continue 
        if noise.sum() == 0: 
            continue
        else:
            (clean_sig, augment_sig), params_i = transforms(signal, noise)
            clean_signals.append(clean_sig)
            augmented_signals.append(augment_sig)
            params.append(params_i)
    clean_signals = torch.cat(clean_signals).unsqueeze(1).float() # add back channel dim
    augmented_signals = torch.cat(augmented_signals).unsqueeze(1).float() # add back channel dim
    params = torch.stack(params) # add back channel dim
    return clean_signals, augmented_signals, params, labels 


def get_rep_wrapped_model(model, input, layer):
    if layer == 'invar_head':
        feature, rep, logits = model(input)
        if len(rep) == 2:
            rep, _ = rep
    elif layer == 'equivar_head':
        feature, (_, rep), logits = model(input)
    else:
        predictions, rep, all_outputs = model(input,  with_latent=True, fake_relu=False)
        rep = all_outputs[layer]
    rep = rep.flatten(start_dim=1)
    return rep

def extract_features_param_decoding(model_1, model_2, loader, layer='avgpool', num_batches=None):
    if num_batches is None:
        num_batches = len(loader)

    responses_clean_1, responses_augmented_1, responses_clean_2, responses_augmented_2,  = [], [], [], []
    params, labels = [], []
    n_ = 0
    for clean_audio, augmented_audio, param, label in tqdm(loader):
        with torch.no_grad():
            clean_audio = clean_audio.cuda()
            augmented_audio = augmented_audio.cuda()
            responses_clean_1.append(get_rep_wrapped_model(model_1, clean_audio, layer))
            responses_augmented_1.append(get_rep_wrapped_model(model_1, augmented_audio, layer))
            responses_clean_2.append(get_rep_wrapped_model(model_2, clean_audio, layer))
            responses_augmented_2.append(get_rep_wrapped_model(model_2, augmented_audio, layer))

            params.append(param)
            labels.append(label)
        n_ += 1
        if n_ == num_batches:
            break

    responses_clean_1 = torch.cat(responses_clean_1)
    responses_augmented_1 = torch.cat(responses_augmented_1)
    responses_clean_2 = torch.cat(responses_clean_2)
    responses_augmented_2 = torch.cat(responses_augmented_2)

    params = torch.cat(params)
    # labels = torch.cat(labels)

    return responses_clean_1, responses_augmented_1, responses_clean_2, responses_augmented_2, params, None


def main(args):
    batch_size = args.batch_size
    jsin_path = '/mnt/ceph/users/jfeather/data/training_datasets_audio/JSIN_all_v3/subsets/'

    train_dataset = jsinV3_precombined_all_signals(root=jsin_path,
                                            train=True,
                                            transform=None,
                                            batch_size=batch_size,
                                            )
    val_dataset = jsinV3_precombined_all_signals(root=jsin_path,
                                            train=False,
                                            transform=None,
                                            batch_size=batch_size,
                                            eval_max=5)

    train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=collate_fn
            )
    test_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=collate_fn
            )

    # Model Loading
    path_model_invariant = '/mnt/ceph/users/igriffith/projects/cochdnn/model_checkpoints/barlow_word_kell2018_base_Matched_blocked_batches_lmbda_1e-2_lr_2e-1_w_augment/checkpoints/epoch=205-step=37080-best_speaker_task.ckpt'
    config_model_invariant_path = Path('model_configs/barlow_word_kell2018_base_Matched_blocked_batches_lmbda_1e-2_lr_2e-1_w_augment.yaml')
    config_model_invariant = yaml.load(open(config_model_invariant_path, 'r'), Loader=yaml.FullLoader)
    invar_model_name = config_model_invariant_path.stem

    path_model_equivariant = '/mnt/ceph/users/igriffith/projects/cochdnn/model_checkpoints/barlow_dualtask_kell2018_base_Matched_blocked_batches_lmbda_1e-2_lr_2e-1_w_augment_eq_lmbda_1e-02/checkpoints/epoch=129-step=23400-best_train.ckpt'
    config_model_equivariant_path = Path('/mnt/ceph/users/igriffith/projects/cochdnn/model_configs/equi_lmbda_search/barlow_dualtask_kell2018_base_Matched_blocked_batches_lmbda_1e-2_lr_2e-1_w_augment_eq_lmbda_1e-02.yaml')
    config_model_equivariant = yaml.load(open(config_model_equivariant_path, 'r'), Loader=yaml.FullLoader)
    equivar_model_name = config_model_equivariant_path.stem

    kell2018_invariant = LitAudioSSL.load_from_checkpoint(config=config_model_invariant, checkpoint_path=path_model_invariant)
    all_layers = kell2018_invariant.metamer_layers
    kell2018_invariant = kell2018_invariant.model.eval().cuda()
    kell2018_equivariant = LitAudioSSL.load_from_checkpoint(config=config_model_equivariant, checkpoint_path=path_model_equivariant).model.eval().cuda()

    layer = all_layers[args.job_id] if args.job_id > -1 else args.layer

    # Extract Features
    rc_test_1, ra_test_1, rc_test_2, ra_test_2, params_test, labels_test = extract_features_param_decoding(
        kell2018_invariant, 
        kell2018_equivariant, 
        test_loader,
        num_batches=args.num_eval,
        layer=layer

    )
    rc_train_1, ra_train_1, rc_train_2, ra_train_2, params_train, labels_train = extract_features_param_decoding(
        kell2018_invariant, 
        kell2018_equivariant, 
        train_loader,
        num_batches=args.num_train,
        layer=layer
    )

    X1_train = torch.cat([rc_train_1, ra_train_1], dim=1).detach().cpu().numpy()
    X2_train = torch.cat([rc_train_2, ra_train_2], dim=1).detach().cpu().numpy()

    X1_test = torch.cat([rc_test_1, ra_test_1], dim=1).detach().cpu().numpy()
    X2_test = torch.cat([rc_test_2, ra_test_2], dim=1).detach().cpu().numpy()

    X1_test = X1_test.reshape(X1_test.shape[0], -1)
    X2_test = X2_test.reshape(X2_test.shape[0], -1)


    Y_train = params_train.detach().cpu().numpy()
    Y_test = params_test.detach().cpu().numpy()

    ## norm parameters 
    # compute global stats 
    Y_global = np.vstack([Y_train, Y_test])
    global_mean = Y_global.mean(0)
    global_std = Y_global.std(0)
    # norm parameters 
    Y_train = (Y_train - global_mean) / global_std
    Y_test  = (Y_test - global_mean) / global_std

    regression_1 = LinearRegression().fit(X1_train, Y_train)
    regression_2 = LinearRegression().fit(X2_train, Y_train)

    score_1 = regression_1.score(X1_test, Y_test)
    score_2 = regression_2.score(X2_test, Y_test)

    preds_1 = regression_1.predict(X1_test)
    preds_2 = regression_2.predict(X2_test)


    # can additionally score on each augmentation individually
    # db_snr, n_semitones, temp_shift, order, low_cutoff, high_cutoff
    AUGMENTATION_LIST = ["dB SNR", "Pitch (semitones)", "% Time warp", "Filter order", "Filter low cutoff", "Filter high cutoff"]

    r2_invariant, r2_equivariant = {}, {}
    for _ in range(len(AUGMENTATION_LIST)):
        r2_invariant[AUGMENTATION_LIST[_]] =  r2_score(Y_test[:, _], preds_1[:, _])
        r2_equivariant[AUGMENTATION_LIST[_]] =  r2_score(Y_test[:, _], preds_2[:, _])


    print(f"Invariant Score: {score_1:.3f}, Equivariant Score: {score_2:.3f}")
    plt.figure(figsize=(15, 5))
    plt.bar(r2_invariant.keys(), r2_invariant.values(), color='blue', alpha=0.5, label='Invariant')
    plt.bar(r2_equivariant.keys(), r2_equivariant.values(), color='red', alpha=0.5, label='Equivariant')
    plt.xticks(rotation=45)
    plt.xlabel('Augmentation')
    plt.ylabel('$R^2$ Score')
    plt.title(f"Invariant model: {invar_model_name}\n Equivariant model: {equivar_model_name}")
    plt.legend()
    fig_out_dir = Path('parameter_decoding') / f"{invar_model_name}_{equivar_model_name}"
    fig_out_dir.mkdir(parents=True, exist_ok=True)
    fig_out_name = fig_out_dir / f"{args.layer}_param_decoding_r2_by_augmentation"
    plt.savefig(fig_out_name, transparent=True, bbox_inches='tight' )



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--batch_size",
        default=192,
        type=int,
        help="Batch size used to extract representations"
    )
    parser.add_argument(
        "--num_train",
        default=50,
        type=int,
        help="Number of training batches to take from training set"
    )
    parser.add_argument(
        "--num_eval",
        default=5,
        type=int,
        help="Number of training batches to take from validation set."
    )
    parser.add_argument(
        "--num_workers",
        default=1,
        type=int,
        help="Number workers per dataloader."
    )
    parser.add_argument(
        "--layer",
        default='avgpool',
        type=str,
        help="Layer to extract representations from"
    )
    parser.add_argument(
        "--job_id",
        default=-1,
        type=int,
        help="Slurm job array index, used to select layers."
    )
    args = parser.parse_args()
    main(args)

