
import torch
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os 
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import logging 
import robustness.audio_functions.audio_transforms as at
from lightning_scripts.jsinV3DataLoader_precombined_batched import jsinV3_precombined_all_signals
import yaml 
from lightning_scripts.lightning_ssl_matched_speech_in_noise import LitAudioSSL
from pathlib import Path 
from argparse import ArgumentParser, BooleanOptionalAction
from lightning import seed_everything

torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

#########################################
# Get stats for augmentation parameters 
#########################################

pearsonr_vec = np.vectorize(pearsonr,
                signature='(n),(n)->(),()') 

def pr2_score(true, pred):
    r, _ = pearsonr_vec(true, pred)
    signs = np.sign(r)
    return signs * np.power(r, 2)

# def functions 
def get_mean_of_uniform(a,b):
    return (a+b) / 2 

def get_std_of_uniform(a,b):
    return np.sqrt(np.square(b-a) / 12.)

def get_std_of_discrete_uniform(a,b):
    return np.sqrt(np.square(b - a + 1 ) / 12.)

def get_uniform_stats(a,b):
    mean = get_mean_of_uniform(a,b)
    std = get_std_of_uniform(a,b)
    return mean, std

def get_discrete_uniform_stats(a,b):
    # mean is same as continuous
    mean = get_mean_of_uniform(a,b)
    # std is different 
    std = get_std_of_discrete_uniform(a,b)
    return mean, std

def get_mean_of_loguniform(a,b):
    return (b - a) / (np.log(b) - np.log(a))

def get_std_of_loguniform(a,b):
    # terms for variance 
    log_diff = np.log(b) - np.log(a)
    numerator = (log_diff) * (np.square(b) - np.square(a)) - (2 * np.square(b-a)) 
    denominator = 2 * np.square(log_diff)
    var = np.divide(numerator, denominator)
    std = np.sqrt(var)
    return std

def get_loguniform_stats(a,b):
    mean = get_mean_of_loguniform(a,b)
    std = get_std_of_loguniform(a,b)
    return mean, std

# Get stats per augmentation
snr_range = [-10, 10]
pitch_range = [-0.5, 0.5]
tempo_range = [-0.8, 1.2]

filter_order_range = [1, 4] # is discrete choice of 1,2,3,4; can treat as discrete uniform over 1-4
range_bandpass_freq_low = [4e1, 4e2]
range_bandpass_freq_high = [4e3, 10e3]

db_snr_mean, db_snr_std = get_uniform_stats(*snr_range)
pitch_mean, pitch_std = get_uniform_stats(*pitch_range)
tempo_mean, tempo_std = get_uniform_stats(*tempo_range)

order_mean, order_std = get_discrete_uniform_stats(*filter_order_range)
low_cutoff_mean, low_cutoff_std = get_loguniform_stats(*range_bandpass_freq_low)
high_cutoff_mean, high_cutoff_std = get_loguniform_stats(*range_bandpass_freq_high)

# group into array for norm 
PARAMS_MEAN = np.array([db_snr_mean, pitch_mean, tempo_mean, order_mean, low_cutoff_mean, high_cutoff_mean],
                           dtype=np.float32)
PARAMS_STD = np.array([db_snr_std, pitch_std, tempo_std, order_std, low_cutoff_std, high_cutoff_std],
                        dtype=np.float32)


######################################
# Define Augmentation stack object
######################################

class dBSNRAugmentation(object):
    def __init__(self, low_db=-10, high_db=10):
        super().__init__()
        self.torch_set_dbSPL = at.DBSPLNormalizeForegroundAndBackground(dbspl=60, use_np=False)
        self.crop = at.CenterCrop(crop_length=40_000) # crop to middle 2 seconds
        self.combine_db_snr = at.CombineWithRandomDBSNRWithParam(low_db, high_db)


    def __call__(self, aud, background):
        logging.getLogger('sox').setLevel(logging.ERROR)

        assert aud is not None, " aud is None on input"
        clean_aud = self.crop(aud)
        assert clean_aud is not None, "clean aud is None post crop"
        # apply augmentations
        # sox aug first to match ssl training 
        assert clean_aud is not None, "clean aud is None post set level"
        clean_aud = torch.from_numpy(clean_aud)
        background = torch.from_numpy(background)
        aug_aud, db_snr = self.combine_db_snr(clean_aud, background)
        aug_aud, _ = self.torch_set_dbSPL(aug_aud, None)
        aug_aud = aug_aud.reshape(1,-1)
        db_SNR_param = torch.tensor([db_snr])
        clean_aud, _ = self.torch_set_dbSPL(clean_aud, None)
        clean_aud = clean_aud.reshape(1,-1)
        return (clean_aud, aug_aud), db_SNR_param
    

class FilterAugmentation(object):
    def __init__(self, low_db=-10, high_db=10):
        super().__init__()
        self.np_set_dbSPL = at.DBSPLNormalizeForegroundAndBackground(dbspl=60, use_np=True)
        self.torch_set_dbSPL = at.DBSPLNormalizeForegroundAndBackground(dbspl=60, use_np=False)
        self.crop = at.CenterCrop(crop_length=40_000) # crop to middle 2 seconds

        self.Pitch = at.ApplySingleAugmentSox('pitch', return_params=True)
        self.Tempo = at.ApplySingleAugmentSox('tempo', return_params=True)
        self.Filter = at.ApplySingleAugmentSox('filter', return_params=True)

    def __call__(self, aud, background):
        logging.getLogger('sox').setLevel(logging.ERROR)

        assert aud is not None, " aud is None on input"
        clean_aud = self.crop(aud)
        assert clean_aud is not None, "clean aud is None post crop"
        clean_aud, _ = self.np_set_dbSPL(clean_aud, None)
        # apply augmentations
        # sox aug first to match ssl training 
        assert clean_aud is not None, "clean aud is None post set level"
        aug_aud, n_semitones =  self.Pitch(clean_aud)
        aug_aud, temp_shift = self.Tempo(aug_aud)
        aug_aud, (order, low_cutoff, high_cutoff) = self.Filter(aug_aud)
        aug_aud = torch.from_numpy(aug_aud)
        aug_aud, _ = self.torch_set_dbSPL(aug_aud, None)
        aug_aud = aug_aud.reshape(1,-1)
        params = torch.tensor([n_semitones, temp_shift, order, low_cutoff, high_cutoff])
        clean_aud = torch.from_numpy(clean_aud).reshape(1,-1)
        return (clean_aud, aug_aud), params

db_snr_transform = dBSNRAugmentation()
filter_transform = FilterAugmentation()


###############################################
# Define collate function for augmentations
###############################################

def snr_collate_fn(batch):
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
   
        else:
            (clean_sig, augment_sig), params_i = db_snr_transform(signal, noise)
            clean_signals.append(clean_sig)
            augmented_signals.append(augment_sig)
            params.append(params_i)
    clean_signals = torch.cat(clean_signals).unsqueeze(1).float() # add back channel dim
    augmented_signals = torch.cat(augmented_signals).unsqueeze(1).float() # add back channel dim
    params = torch.stack(params) # add back channel dim
    return clean_signals, augmented_signals, params, labels 

def filter_collate_fn(batch):
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
    for (signal, _) in  zip(*batch[:2]):
        if signal.sum() == 0 or signal is None:
            continue 
   
        else:
            (clean_sig, augment_sig), params_i = filter_transform(signal, None)
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
    if num_batches is None or num_batches == -1:
        num_batches = len(loader)

    responses_clean_1, responses_augmented_1, responses_clean_2, responses_augmented_2,  = [], [], [], []
    params, labels = [], []
    n_ = 0
    with torch.no_grad():
        for clean_audio, augmented_audio, param, label in tqdm(loader):
            clean_audio = clean_audio.cuda()
            augmented_audio = augmented_audio.cuda()
            responses_clean_1.append(get_rep_wrapped_model(model_1, clean_audio, layer).cpu())
            responses_augmented_1.append(get_rep_wrapped_model(model_1, augmented_audio, layer).cpu())
            responses_clean_2.append(get_rep_wrapped_model(model_2, clean_audio, layer).cpu())
            responses_augmented_2.append(get_rep_wrapped_model(model_2, augmented_audio, layer).cpu())

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
    jsin_path = str(require_path(JSIN_PATH, 'COCHDNN_JSIN_DIR', 'JSIN/WSN dataset'))

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

    snr_train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=snr_collate_fn
            )

    filter_train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=filter_collate_fn
            )

    snr_test_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=snr_collate_fn
            )

    filter_test_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=filter_collate_fn
            )

    # Model Loading
    config_model_invariant_path = Path(args.invar_model_config) # Path('model_configs/barlow_word_kell2018_base_Matched_blocked_batches_lmbda_1e-2_lr_2e-1_w_augment.yaml')
    config_model_invariant = yaml.load(open(config_model_invariant_path, 'r'), Loader=yaml.FullLoader)

    if args.invar_model_ckpt == '':
        checkpoint_dir = Path(args.exp_dir) / f"{config_model_invariant_path.stem}/checkpoints"
        # Get latest ckpt for equivariant model 
        path_model_invariant = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getctime)[-1]
    else:
        path_model_invariant = args.invar_model_ckpt  
    print('Invariant checkpoint path: ', path_model_invariant)

    invar_model_name = config_model_invariant_path.stem


    if args.config_list:
        with open(args.config_list, 'rb') as f:
            model_config = pickle.load(f)
            config_model_equivariant_path = Path(model_config[args.config_id])
    else:
        config_model_equivariant_path = Path(args.equi_model_config)
   
    config_model_equivariant = yaml.load(open(config_model_equivariant_path, 'r'), Loader=yaml.FullLoader)

    if args.equi_model_ckpt == '':
        checkpoint_dir = Path(args.exp_dir) / f"{config_model_equivariant_path.stem}/checkpoints"
        # Get latest ckpt for equivariant model 
        path_model_equivariant = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getctime)[-1]
    else:
        path_model_equivariant = args.equi_model_ckpt  
    print('Equivariant checkpoint path: ', path_model_equivariant)

    equivar_model_name = config_model_equivariant_path.stem


    print(f"Running models: {invar_model_name} and {equivar_model_name}")

    invariant_model = LitAudioSSL.load_from_checkpoint(config=config_model_invariant, checkpoint_path=path_model_invariant)
    all_layers = invariant_model.metamer_layers
    invariant_model = invariant_model.model.eval().cuda()
    equivariant_model = LitAudioSSL.load_from_checkpoint(config=config_model_equivariant, checkpoint_path=path_model_equivariant).model.eval().cuda()

    layer = all_layers[args.job_id] if args.job_id > -1 else args.layer

    # set seed for replicable sampling
    seed_everything(0)
    # Extract Features for dB SNR decoding 
    rc_test_1, ra_test_1, rc_test_2, ra_test_2, params_test, labels_test = extract_features_param_decoding(
        invariant_model, 
        equivariant_model, 
        snr_test_loader,
        num_batches=args.num_eval,
        layer=layer

    )
    rc_train_1, ra_train_1, rc_train_2, ra_train_2, params_train, labels_train = extract_features_param_decoding(
        invariant_model, 
        equivariant_model, 
        snr_train_loader,
        num_batches=args.num_train,
        layer=layer
    )

    X1_train = torch.cat([rc_train_1, ra_train_1], dim=1).detach().cpu().numpy()
    X2_train = torch.cat([rc_train_2, ra_train_2], dim=1).detach().cpu().numpy()

    X1_test = torch.cat([rc_test_1, ra_test_1], dim=1).detach().cpu().numpy()
    X2_test = torch.cat([rc_test_2, ra_test_2], dim=1).detach().cpu().numpy()

    X1_test = X1_test.reshape(X1_test.shape[0], -1)
    X2_test = X2_test.reshape(X2_test.shape[0], -1)


    ## Cut params to just be db SNR
    Y_train_db_SNR = params_train[:,0].detach().cpu().numpy()
    Y_test_db_SNR = params_test[:,0].detach().cpu().numpy()

    # get valid example ixs (where SNR is not inf)
    train_IXS = np.argwhere(~np.isinf(Y_train_db_SNR)).flatten()
    test_IXS = np.argwhere(~np.isinf(Y_test_db_SNR)).flatten()

    X1_train = X1_train[train_IXS]
    X1_test = X1_test[test_IXS]

    X2_train = X2_train[train_IXS]
    X2_test = X2_test[test_IXS]

    Y_train_db_SNR = Y_train_db_SNR[train_IXS].reshape(-1,1)
    Y_test_db_SNR = Y_test_db_SNR[test_IXS].reshape(-1,1)

    ## norm parameters 
    # use emperical stats 
    db_snr_mean = Y_train_db_SNR.mean(0)
    db_snr_std = Y_train_db_SNR.std(0)
    Y_train_db_SNR = (Y_train_db_SNR - db_snr_mean) / db_snr_std
    Y_test_db_SNR  = (Y_test_db_SNR - db_snr_mean) / db_snr_std

    regression_1_db_snr = Ridge(alpha=args.ridge_alpha).fit(X1_train, Y_train_db_SNR)
    regression_2_db_snr = Ridge(alpha=args.ridge_alpha).fit(X2_train, Y_train_db_SNR)

    score_1_db_snr = regression_1_db_snr.score(X1_test, Y_test_db_SNR)
    score_2_db_snr = regression_2_db_snr.score(X2_test, Y_test_db_SNR)

    preds_1_db_snr = regression_1_db_snr.predict(X1_test).reshape(-1,1)
    preds_2_db_snr = regression_2_db_snr.predict(X2_test).reshape(-1,1)

    # set seed for replicable sampling
    seed_everything(0)

    # Extract Features for Filter param decoding 
    rc_test_1_filter, ra_test_1_filter, rc_test_2_filter, ra_test_2_filter, params_test_filter, labels_test_filter = extract_features_param_decoding(
        invariant_model, 
        equivariant_model, 
        filter_test_loader,
        num_batches=args.num_eval,
        layer=layer
    )
    rc_train_1_filter, ra_train_1_filter, rc_train_2_filter, ra_train_2_filter, params_train_filter, labels_train_filter = extract_features_param_decoding(
        invariant_model, 
        equivariant_model, 
        filter_train_loader,
        num_batches=args.num_train,
        layer=layer
    )

    X1_train_filter = torch.cat([rc_train_1_filter, ra_train_1_filter], dim=1).detach().cpu().numpy()
    X2_train_filter = torch.cat([rc_train_2_filter, ra_train_2_filter], dim=1).detach().cpu().numpy()

    X1_test_filter = torch.cat([rc_test_1_filter, ra_test_1_filter], dim=1).detach().cpu().numpy()
    X2_test_filter = torch.cat([rc_test_2_filter, ra_test_2_filter], dim=1).detach().cpu().numpy()

    X1_test_filter = X1_test_filter.reshape(X1_test_filter.shape[0], -1)
    X2_test_filter = X2_test_filter.reshape(X2_test_filter.shape[0], -1)


    ## Cut params to just be db SNR
    Y_train_filter = params_train_filter.detach().cpu().numpy()
    Y_test_filter = params_test_filter.detach().cpu().numpy()

    ## norm parameters 
    # Try emperical means 
    filter_mean = Y_train_filter.mean(0)
    filter_std = Y_train_filter.std(0)
    Y_train_filter = (Y_train_filter - filter_mean) / filter_std
    Y_test_filter  = (Y_test_filter - filter_mean) / filter_std

    regression_1_filter = Ridge(alpha=args.ridge_alpha).fit(X1_train_filter, Y_train_filter)
    regression_2_filter = Ridge(alpha=args.ridge_alpha).fit(X2_train_filter, Y_train_filter)

    score_1_filter = regression_1_filter.score(X1_test_filter, Y_test_filter)
    score_2_filter = regression_2_filter.score(X2_test_filter, Y_test_filter)

    preds_1_filter = regression_1_filter.predict(X1_test_filter)
    preds_2_filter = regression_2_filter.predict(X2_test_filter)

    # can additionally score on each augmentation individually
    # db_snr, n_semitones, temp_shift, order, low_cutoff, high_cutoff
    AUGMENTATION_LIST = ["dB SNR", "Pitch (semitones)", "% Time warp", "Filter order", "Filter low cutoff", "Filter high cutoff"]

    r2_invariant, r2_equivariant = {}, {}
    for _ in range(len(AUGMENTATION_LIST)):
        if _ == 0:
            r2_invariant[AUGMENTATION_LIST[_]] =  r2_score(Y_test_db_SNR[:, _], preds_1_db_snr[:, _])
            r2_equivariant[AUGMENTATION_LIST[_]] =  r2_score(Y_test_db_SNR[:, _], preds_2_db_snr[:, _])
        else:
            r2_invariant[AUGMENTATION_LIST[_]] =  r2_score(Y_test_filter[:, _ - 1], preds_1_filter[:, _ - 1])
            r2_equivariant[AUGMENTATION_LIST[_]] =  r2_score(Y_test_filter[:, _ - 1], preds_2_filter[:, _ - 1])
    
    
    ## Get additional metric 
    r2_invariant_pr2, r2_equivariant_pr2 = {}, {}
    for _ in range(len(AUGMENTATION_LIST)):
        if _ == 0:
            r2_invariant_pr2[AUGMENTATION_LIST[_]] =  pr2_score(Y_test_db_SNR[:, _], preds_1_db_snr[:, _])
            r2_equivariant_pr2[AUGMENTATION_LIST[_]] =  pr2_score(Y_test_db_SNR[:, _], preds_2_db_snr[:, _])
        else:
            r2_invariant_pr2[AUGMENTATION_LIST[_]] =  pr2_score(Y_test_filter[:, _-1], preds_1_filter[:, _-1])
            r2_equivariant_pr2[AUGMENTATION_LIST[_]] =  pr2_score(Y_test_filter[:, _-1], preds_2_filter[:, _-1])


    print(f"Invariant Scores:, Equivariant Scores: ")
    print(f"{score_1_db_snr:.3f} (R^2 dB SNR), {score_2_db_snr:.3f} (R^2 dB SNR)")
    print(f"{score_1_filter:.3f} (R^2 filter), {score_2_filter:.3f} (R^2 filter)")

    ridge_title_str = f"Ridge $\\alpha=${args.ridge_alpha:.1f}" if args.ridge_alpha != 0 else ''
    plt.figure(figsize=(10, 5))
    bar_width = 0.25
    bar_x = np.arange(len(r2_invariant.keys()))
    plt.bar(bar_x, r2_invariant.values(), color='blue', alpha=0.5, label='Invariant', width=bar_width)
    plt.bar(bar_x + bar_width, r2_equivariant.values(), color='red', alpha=0.5, label='Equivariant', width=bar_width)
    plt.axhline(0, color='k', lw=0.5)
    plt.xticks(rotation=45, ticks=bar_x + (bar_width * 0.5), labels=list(r2_invariant.keys()))
    plt.xlabel('Augmentation')
    plt.ylabel('$R^2$ Score')
    plt.ylim(-1.1,1.1)
    plt.title(f"Invariant model: {invar_model_name}\n Equivariant model: {equivar_model_name}\nLayer: {layer}\n{ridge_title_str}")
    plt.legend()
    fig_out_dir = Path('parameter_decoding') / f"{invar_model_name}_{equivar_model_name}"
    fig_out_dir.mkdir(parents=True, exist_ok=True)
    fig_out_name = fig_out_dir / f"{layer}_param_decoding_r2_by_augmentation{f"_{args.ridge_alpha:.0e}"if args.ridge_alpha != 0.0 else ''}"
    plt.savefig(fig_out_name, transparent=False, bbox_inches='tight' )

    plt.figure(figsize=(10, 5))
    bar_width = 0.25
    bar_x = np.arange(len(r2_invariant.keys()))
    plt.bar(bar_x, r2_invariant_pr2.values(), color='blue', alpha=0.5, label='Invariant', width=bar_width)
    plt.bar(bar_x + bar_width, r2_equivariant_pr2.values(), color='red', alpha=0.5, label='Equivariant', width=bar_width)
    plt.axhline(0, color='k', lw=0.5)
    plt.xticks(rotation=45, ticks=bar_x + (bar_width * 0.5), labels=list(r2_invariant.keys()))
    plt.xlabel('Augmentation')
    plt.ylabel("Pearson's $r^2$ Score")
    plt.ylim(-1.1,1.1)
    plt.title(f"Invariant model: {invar_model_name}\n Equivariant model: {equivar_model_name}\nLayer: {layer}\n{ridge_title_str}")
    plt.legend()
    fig_out_dir.mkdir(parents=True, exist_ok=True)
    fig_out_name = fig_out_dir / f"{layer}_param_decoding_pearsons_r2_by_augmentation{f"_{args.ridge_alpha:.0e}"if args.ridge_alpha != 0.0 else ''}"
    plt.savefig(fig_out_name, transparent=False, bbox_inches='tight' )

    # save results 
    data_out_name = fig_out_dir / f"{layer}_r2_decoding_values{f"_ridge_alpha_{args.ridge_alpha:.0e}"if args.ridge_alpha != 0.0 else ''}.pkl"
    data = dict(r2_invariant=r2_invariant, r2_equivariant=r2_equivariant,
                 pr2_invariant=r2_invariant_pr2, pr2_equivariant=r2_equivariant_pr2)

    with open(data_out_name, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)



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
    parser.add_argument(
        "--config_id",
        default=-1,
        type=int,
        help="Slurm job array index, used to select equivariant model."
    )
    parser.add_argument(
        "--exp_dir",
        default=Path("./model_checkpoints"),
        type=Path,
        help="Directory to save checkpoints and logs to. (Default: './exp')",
    )
    parser.add_argument(
        "--invar_model_config",
        default=Path("model_configs/barlow_word_kell2018_base_Matched_blocked_batches_lmbda_1e-2_lr_2e-1_w_augment.yaml"),
        type=Path,
        help="Path to invariant model config",
    )
    parser.add_argument(
        "--invar_model_ckpt",
        default='', # Path("model_checkpoints/barlow_word_kell2018_base_Matched_blocked_batches_lmbda_1e-2_lr_2e-1_w_augment/checkpoints/epoch=205-step=37080-best_speaker_task.ckpt"),
        type=str,
        help="Path to invariant model checkpoint",
    )
    parser.add_argument(
        "--equi_model_config",
        default=Path("model_configs/barlow_dualtask_kell2018_base_Matched_blocked_batches_lmbda_1e-2_lr_2e-1_w_augment_eq_lmbda_1e-1_fixed_loss.yaml"),
        type=Path,
        help="Path to equivariant model config",
    )
    parser.add_argument(
        "--equi_model_ckpt",
        default='', # Path("model_checkpoints/barlow_word_kell2018_base_Matched_blocked_batches_lmbda_1e-2_lr_2e-1_w_augment/checkpoints/epoch=205-step=37080-best_speaker_task.ckpt"),
        type=str,
        help="Path to equivariant model checkpoint",
    )
    parser.add_argument('--config_list', type=str, help='Path to list of config files.')
    parser.add_argument(
        "--ridge_alpha",
        default=0.0,
        type=float,
        help="Alpha to use in ridge regression. Default (0) is same as OLS."
    )
    args = parser.parse_args()
    main(args)

