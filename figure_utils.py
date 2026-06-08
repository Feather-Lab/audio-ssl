import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Standard model ordering and colors for consistent plotting across notebooks
STANDARD_HUE_ORDER = [
    # Supervisedp
    'CochCNN9 supervised multi-task',
    'CochCNN9 supervised word',
    'CochCNN9 supervised audioset',
    'CochCNN9 scaled supervised',
    'CochCNN9 scaled aud. event',
    # SSL / Equivariant
    'CochCNN9 ssl λ=0.0',
    'CochCNN9 ssl λ=0.1',
    'CochCNN9 ssl λ=0.2',
    'CochCNN9 ssl λ=0.3',
    'CochCNN9 ssl λ=0.4',
    'CochCNN9 ssl λ=0.5',
    'CochCNN9 scaled ssl λ=0.0',
    'CochCNN9 scaled ssl λ=0.5',
    # SSL / Dual
    'CochCNN9 ssl dual λ=0.0',
    'CochCNN9 ssl dual λ=0.0 unpaired augs',
    'CochCNN9 ssl dual λ=0.01',
    'CochCNN9 ssl dual λ=0.1',
    'CochCNN9 ssl dual λ=0.5',
    # Resnets 50
    'resnet50 ssl λ=0.5',
    'resnet50 scaled ssl λ=0.5',
    # Resnet 18
    'resnet18 scaled ssl λ=0.0',
    'resnet18 scaled ssl λ=0.5',
    # BYOL-A
    'byol-a',
]

STANDARD_BASE_COLORS = {
    'CochCNN9 supervised word': 'blue',
    'CochCNN9 supervised audioset': 'orange',
    'CochCNN9 scaled supervised': 'gold',
    'CochCNN9 scaled aud. event': 'gold',
    'CochCNN9 supervised multi-task': 'grey',
}

def get_x_labels(hue_order):
    x_labels = []
    for name in hue_order:
        label = model_label(name)
        if 'scaled ssl' in name:
            if label.startswith('λ='):
                lambda_val = label.replace('λ=', '')
                x_labels.append(f"scaled ssl λ={lambda_val}")
            else:
                x_labels.append(f"scaled ssl {label}")
        elif 'ssl' in name and 'scaled' not in name:
            if label.startswith('λ='):
                lambda_val = label.replace('λ=', '')
                x_labels.append(f"ssl λ={lambda_val}")
            else:
                x_labels.append(f"ssl {label}")
        else:
            x_labels.append(label)
    return x_labels



def get_standard_hue_order():
    """Get the standard model order for plots, normalized."""
    return normalize_model_list(STANDARD_HUE_ORDER)


def get_standard_base_colors():
    """Get the standard base colors for supervised models, normalized."""
    return normalize_palette_dict(STANDARD_BASE_COLORS)


def filter_models_for_plot(df, hue_order=None):
    """
    Filter a results dataframe to only include models in the standard plot order.
    Used so bar plots show only the models we define in STANDARD_HUE_ORDER.

    Args:
        df: DataFrame with a 'model_name' column (e.g. concatenated ESC-50, speech, word, nsynth).
        hue_order: List of model names to keep. If None, uses get_standard_hue_order().

    Returns:
        DataFrame subset of df where model_name is in hue_order.
    """
    if hue_order is None:
        hue_order = get_standard_hue_order()
    return df[df['model_name'].isin(hue_order)].copy()


def get_model_groups(hue_order=None):
    """
    Get model groups (supervised, SSL, scaled SSL, BYOL-A) from hue_order.
    
    Args:
        hue_order: Optional normalized hue_order. If None, uses standard.
    
    Returns:
        Tuple of (supervised_names, ssl_names, scaled_ssl_names, byol_names)
    """
    if hue_order is None:
        hue_order = get_standard_hue_order()
    
    supervised_names = [
        'CochCNN9 supervised multi-task',
        'CochCNN9 supervised word',
        'CochCNN9 supervised audioset',
        'CochCNN9 scaled supervised',
    ]
    ssl_names = [
        name for name in hue_order
        if name.startswith('CochCNN9 ssl ') and 'dual' not in name and 'scaled' not in name
    ]
    scaled_ssl_names = [
        name for name in hue_order if name.startswith('CochCNN9 scaled ssl ')
    ]
    byol_names = ['byol-a']
    
    supervised_names = normalize_model_list(supervised_names)
    ssl_names = normalize_model_list(ssl_names)
    scaled_ssl_names = normalize_model_list(scaled_ssl_names)
    byol_names = normalize_model_list(byol_names)
    
    return supervised_names, ssl_names, scaled_ssl_names, byol_names


def get_plot_groups(hue_order=None):
    """
    Get plot groups for grouped bar plots.
    
    Args:
        hue_order: Optional normalized hue_order. If None, uses standard.
    
    Returns:
        List of tuples: [('Supervised', supervised_names), ('SSL', ssl_names), ...]
    """
    supervised_names, ssl_names, scaled_ssl_names, byol_names = get_model_groups(hue_order)
    
    return [
        ('Supervised', supervised_names),
        ('SSL', ssl_names),
        ('Scaled\nSSL', scaled_ssl_names),
        ('BYOL-A', byol_names),
    ]
    
def set_bar_labels(axs,
                   bars,
                   plot_names,
                   white_text_substrs=["word", "CE"],
                   ymin_bars=0.49,
                   no_olap_pad=0.0225,
                   inside_fit_tolerance=0.02,
                   fontsize=12,
                   sem_vals=None,
                   y_cut_for_olap=None,
                   x_shift_size=0.0,
    ):

    fig = axs.get_figure()
    if y_cut_for_olap is None:
        if fig.canvas is not None:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
        else:
            renderer = None

    for ix, bar in enumerate(bars):
        bar_height = bar.get_height()
        bar_name = plot_names[ix].replace('supervised', 'sup.')

        if y_cut_for_olap is not None:
            if bar_height > y_cut_for_olap:
                y = ymin_bars
            else:
                y = bar_height + no_olap_pad
            text_color = "k"
            if white_text_substrs and any(sub_str in bar_name for sub_str in white_text_substrs):
                text_color = "white"
            x = x_shift_size + (bar.get_x() + bar.get_width() / 2.)
        else:
            bar_interior = bar_height - ymin_bars
            if renderer is not None:
                tmp_txt = axs.text(0, 0, bar_name, rotation=90, fontsize=fontsize, alpha=0)
                bbox = tmp_txt.get_window_extent(renderer=renderer)
                tmp_txt.remove()
                y0 = axs.transData.inverted().transform((0, 0))[1]
                y1 = axs.transData.inverted().transform((0, bbox.height))[1]
                label_height_data = y1 - y0
            else:
                label_height_data = len(bar_name) * 0.01

            if label_height_data <= (bar_interior + inside_fit_tolerance):
                y = ymin_bars
                text_color = "k"
                if white_text_substrs and any(sub_str in bar_name for sub_str in white_text_substrs):
                    text_color = "white"
            else:
                top = bar_height + (sem_vals[ix] if sem_vals is not None else 0.0)
                y = top + no_olap_pad
                text_color = "k"
            x = bar.get_x() + bar.get_width() / 2.

        axs.text(
            x,
            y,
            bar_name,
            ha='center',
            va="bottom",
            rotation=90,
            color=text_color,
            fontsize=fontsize,
        )

def strip_cochdnn9(name):
    """Remove 'CochCNN9 ' prefix from model name."""
    return name.replace('CochCNN9 ', '')


def lambda_label(name):
    """Extract lambda value from model name for label."""
    match = re.search(r'(?:λ|lambda)\s*=\s*([0-9.]+)', name, re.IGNORECASE)
    return f"λ={match.group(1)}" if match else strip_cochdnn9(name)


def model_label(name, with_ssl=False, use_ssl_names=False):
    """
    Format model name for plot labels.
    
    Parameters
    ----------
    name : str
        Model name to format
    with_ssl : bool
        If True, prefix with 'ssl' or 'scaled ssl'
    use_ssl_names : bool
        If True, use descriptive SSL names:
        - λ=0.0 → iSSL (invariant SSL)
        - λ=0.5 → CE-SSL (contrastive-equivariant SSL)
        With "scaled" prefix for scaled versions.
        If False (default), use lambda notation (λ=X.X) for all values.
    """
    is_scaled = 'scaled ssl' in name
    is_ssl = 'ssl' in name
    
    if "ssl" in name:
        # Extract lambda value
        match = re.search(r'(?:λ|lambda)\s*=\s*([0-9.]+)', name, re.IGNORECASE)
        if match:
            lambda_val = match.group(1)
            
            if use_ssl_names:
                # Map lambda values to descriptive names
                if lambda_val in ['0.0', '0']:
                    label = 'iSSL'
                elif lambda_val in ['0.5']:
                    label = 'CE-SSL'
                else:
                    label = f'λ={lambda_val}'
                
                # Add scaled prefix if applicable
                if is_scaled:
                    label = f'scaled {label}'
            else:
                # Use lambda notation
                label = f'λ={lambda_val}'
        else:
            label = strip_cochdnn9(name)
    else:
        label = strip_cochdnn9(name)
    
    if with_ssl:
        if is_scaled:
            label = f'scaled ssl {label}'
        elif is_ssl:
            label = f'ssl {label}'

    # Display label for supervised audioset
    label = label.replace('supervised audioset', 'supervised aud. events')
    label = label.replace('scaled supervised', "scaled aud. event")
    label = label.replace('scaled sup', "scaled aud. event")
    return label


def normalize_model_name(model_name):
    name = model_name
    name = re.sub(r'\bkell2018\b', 'CochCNN9', name, flags=re.IGNORECASE)
    name = re.sub(r'\bCochDNN9\b', 'CochCNN9', name)  # Handle legacy naming

    name = re.sub(r'\bscaled ssl\b\s+eq\b', 'scaled ssl', name, flags=re.IGNORECASE)
    name = re.sub(r'\bssl\b\s+eq\b', 'ssl', name, flags=re.IGNORECASE)

    # Apply SSL normalization rules to all architectures (CochCNN9, resnet18, resnet50, etc.)
    # Match any architecture prefix followed by ssl word/audioset
    name = re.sub(r'\b(CochCNN9|resnet\d+)\b\s+ssl\s+word\b', r'\1 ssl λ=0.0', name, flags=re.IGNORECASE)
    name = re.sub(r'\b(CochCNN9|resnet\d+)\b\s+ssl\s+audioset\b', r'\1 ssl λ=1.0', name, flags=re.IGNORECASE)

    if 'supervised' in name and 'jsin' in name and 'audioset' in name:
        name = re.sub(
            r'\bsupervised\b\s+jsin\s+audioset',
            'supervised audioset',
            name,
            flags=re.IGNORECASE,
        )

    if 'unbalanced audioset' in name:
        if 'supervised' in name:
            name = re.sub(
                r'\bsupervised\b\s+unbalanced audioset',
                'scaled aud. event',
                name,
                flags=re.IGNORECASE,
            )
        else:
            name = re.sub(
                r'\bssl\b(?:\s+dual)?\s+unbalanced audioset',
                'scaled ssl',
                name,
                flags=re.IGNORECASE,
            )
        name = re.sub(r'\s+', ' ', name).strip()

    return name


def normalize_model_list(model_names):
    return [normalize_model_name(name) for name in model_names]


def normalize_palette_dict(hue_dict):
    return {normalize_model_name(k): v for k, v in hue_dict.items()}


def marker_palette_for_hue(hue_order):
    return ['s' if 'supervised' in model else 'o' for model in hue_order]


def _parse_lambda_value(name):
    match = re.search(r'λ=([0-9.]+)', name)
    return float(match.group(1)) if match else None


# Fixed lambda values for consistent color mapping across all plots
_FIXED_LAMBDA_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def _palette_for_lambdas(hue_order, prefix, palette_name):
    names = [n for n in hue_order if n.startswith(prefix) and 'λ=' in n]
    if not names:
        return {}
    # Use fixed lambda values for consistent colors across all plots
    colors = sns.color_palette(palette_name, n_colors=len(_FIXED_LAMBDA_VALUES))
    color_map = dict(zip(_FIXED_LAMBDA_VALUES, colors))
    return {n: color_map[_parse_lambda_value(n)] for n in names if _parse_lambda_value(n) is not None and _parse_lambda_value(n) in color_map}


def _palette_for_lambdas_matching(hue_order, match_fn, palette_name):
    names = [n for n in hue_order if match_fn(n) and 'λ=' in n]
    if not names:
        return {}
    # Use fixed lambda values for consistent colors across all plots
    # This ensures the same model always gets the same color regardless of
    # which other models are included in the current plot's hue_order
    colors = sns.color_palette(palette_name, n_colors=len(_FIXED_LAMBDA_VALUES))
    color_map = dict(zip(_FIXED_LAMBDA_VALUES, colors))
    result = {}
    for n in names:
        lambda_val = _parse_lambda_value(n)
        if lambda_val is not None and lambda_val in color_map:
            result[n] = color_map[lambda_val]
    return result


def build_model_palette(hue_order, base_colors=None):
    base = dict(base_colors or {})
    palette = {}
    
    # Initialize palette with base colors for models that have them
    for name in hue_order:
        if name in base:
            palette[name] = base[name]

    # Handle resnet SSL models (with lambda) - must come before general SSL check
    # Check for ' ssl ' with spaces to avoid matching 'scaled ssl'
    resnet_ssl = _palette_for_lambdas_matching(
        hue_order, 
        lambda n: n.startswith('resnet') and ' ssl ' in n and 'scaled ssl' not in n and 'ssl dual' not in n,
        'RdPu'
    )
    palette.update(resnet_ssl)
    
    # Handle general SSL models (excluding resnet and already handled models)
    general_ssl = _palette_for_lambdas_matching(
        hue_order,
        lambda n: ' ssl ' in n and 'scaled ssl' not in n and 'ssl dual' not in n and not n.startswith('resnet'),
        'Greens',
    )
    palette.update(general_ssl)
    
    # Handle scaled SSL models - separate by architecture
    # Scaled SSL for CochCNN9 - use Purples
    cochdnn9_scaled_ssl = _palette_for_lambdas_matching(
        hue_order, 
        lambda n: 'scaled ssl' in n and n.startswith('CochCNN9'),
        'Purples'
    )
    palette.update(cochdnn9_scaled_ssl)
    
    # Scaled SSL for resnet - use YlOrRd (yellow-orange-red) to differentiate from Purples
    resnet_scaled_ssl = _palette_for_lambdas_matching(
        hue_order, 
        lambda n: 'scaled ssl' in n and n.startswith('resnet'),
        'YlOrRd'
    )
    palette.update(resnet_scaled_ssl)
    
    # Handle SSL dual models
    ssl_dual = _palette_for_lambdas_matching(
        hue_order, 
        lambda n: 'ssl dual' in n,
        'Blues'
    )
    palette.update(ssl_dual)
    
    # Handle supervised CochCNN9 models that aren't in base_colors
    cochdnn9_supervised = [n for n in hue_order if n.startswith('CochCNN9') and 'supervised' in n and n not in palette]
    if cochdnn9_supervised:
        # Use a different palette for supervised CochCNN9 models not in base_colors
        colors = sns.color_palette('Blues', n_colors=max(len(cochdnn9_supervised), 2))
        for i, name in enumerate(cochdnn9_supervised):
            palette[name] = colors[i % len(colors)]
    
    # Handle supervised resnet models (they don't have lambda, so need special handling)
    resnet_supervised = [n for n in hue_order if n.startswith('resnet') and 'supervised' in n and n not in palette]
    if resnet_supervised:
        # Use a different palette for supervised resnet models
        colors = sns.color_palette('Oranges', n_colors=max(len(resnet_supervised), 2))
        for i, name in enumerate(resnet_supervised):
            palette[name] = colors[i % len(colors)]
    
    # Handle any other resnet models that don't have lambda (fallback)
    resnet_other = [n for n in hue_order if n.startswith('resnet') and n not in palette]
    if resnet_other:
        colors = sns.color_palette('RdPu', n_colors=max(len(resnet_other), 2))
        for i, name in enumerate(resnet_other):
            palette[name] = colors[i % len(colors)]
    
    # Handle any other CochCNN9 models that don't have lambda (fallback)
    cochdnn9_other = [n for n in hue_order if n.startswith('CochCNN9') and n not in palette]
    if cochdnn9_other:
        colors = sns.color_palette('Greens', n_colors=max(len(cochdnn9_other), 2))
        for i, name in enumerate(cochdnn9_other):
            palette[name] = colors[i % len(colors)]

    # Keep AudioMAE models visually distinct and consistent.
    audiomae_models = [
        n for n in hue_order
        if ('audiomae' in n.lower()) or ('audio mae' in n.lower()) or ('audio-mae' in n.lower())
    ]
    for name in audiomae_models:
        palette[name] = 'cyan'

    # Ensure ALL models in hue_order have a color (final fallback to gray)
    # This is the absolute final check - every model MUST have a color
    for name in hue_order:
        if name not in palette:
            palette[name] = '#444444'
    
    # Final validation: ensure palette has all keys from hue_order
    missing = set(hue_order) - set(palette.keys())
    if missing:
        # This should never happen, but if it does, assign gray to missing models
        for name in missing:
            palette[name] = '#444444'
    
    # Double-check: ensure we have exactly the right keys
    # Add any missing keys and remove any extra keys (though we shouldn't have extra)
    final_palette = {}
    for name in hue_order:
        final_palette[name] = palette.get(name, '#444444')
    
    return final_palette


def format_model_str(path):
    path_str = str(path)

    arch_match = re.search(r'(kell2018|resnet\d+|byol-a|byola)', path_str, re.IGNORECASE)
    arch = arch_match.group(1) if arch_match else 'unknown'

    ssl_keywords = ['barlow', 'mmcr', 'invariant', 'equivariant']
    is_ssl = any(k in path_str for k in ssl_keywords)

    is_dual = 'dual' in path_str or 'dualtask' in path_str

    modifiers = []
    if 'word_speaker_audioset' in path_str:
        modifiers.append('multi-task')
    elif 'word_speaker' in path_str:
        modifiers.append('word speaker')
    elif 'word' in path_str:
        modifiers.append('word')
    elif 'jsin' in path_str:
        modifiers.append('audioset')
    elif 'audioset' in path_str and "unbalanced" not in path_str and "only" not in path_str:
        modifiers.append('jsin audioset')
    elif 'audioset' in path_str and ("unbalanced" in path_str or "only" in path_str):
        modifiers.append('unbalanced audioset')

    eq_lambda = ''
    if 'equivariant' in path_str or 'eq_lmbda' in path_str:
        lambda_match = re.search(r'eq_lmbda_([-\d.e]+)', path_str)
        if lambda_match:
            eq_lambda = str(float(lambda_match.group(1)))
    # For resnet models with invariant_only, set lambda to 0.0 if no eq_lmbda found
    elif 'invariant_only' in path_str and arch.startswith('resnet'):
        eq_lambda = '0.0'
    
    

    task_parts = []
    if is_ssl:
        if is_dual:
            task_parts.append('ssl dual')
        elif eq_lambda:
            task_parts.append('ssl eq')
        else:
            task_parts.append('ssl')

        if 'multi-task' not in modifiers:
            task_parts.extend(modifiers)
    else:
        task_parts.append('supervised')
        task_parts.extend(modifiers)

    if eq_lambda:
        task_parts.append(f'λ={eq_lambda}')

    task = ' '.join(task_parts)

    aug_str = ''
    if 'paired_augmentations' in path_str:
        aug_str = 'paired augmentations'
    elif 'no_shared_fg_augments' in path_str:
        aug_str = 'unpaired augs'

    test_layer = Path(path).stem.split("_")[-1]

    if arch and arch.lower() == 'byola':
        if 'invariant' in path_str:
            model_name = 'byol-a λ=0.0'
        else:
            model_name = 'byol-a'
        return normalize_model_name(model_name), arch, test_layer, eq_lambda

    model_name = f"{arch} {task}"
    if aug_str:
        model_name = f"{model_name} {aug_str}"

    return normalize_model_name(model_name), arch, test_layer, eq_lambda


def format_model_str_word_task(path):
    full_path_str = str(path)
    path_str = Path(path).stem
    path_root = path_str.split('_linear_eval')[0]

    arch_match = re.search(r'(kell2018|resnet\d+|byol-a|byola)', path_root, re.IGNORECASE)
    arch = arch_match.group(1) if arch_match else 'unknown'

    ssl_keywords = ['barlow', 'mmcr', 'invariant', 'equivariant']
    is_ssl = any(k in path_root for k in ssl_keywords)

    is_dual = 'dual' in path_root or 'dualtask' in path_root

    modifiers = []
    if 'word_speaker_audioset' in path_root:
        modifiers.append('multi-task')
    elif 'word' in path_root:
        modifiers.append('word')
    elif 'jsin' in path_root:
        modifiers.append('audioset')
    elif 'audioset' in path_root and ("unbalanced" not in path_root and "only" not in path_root):
        modifiers.append('jsin audioset')
    elif 'audioset' in path_root and ("unbalanced" in path_root or "only" in path_root):
        modifiers.append('unbalanced audioset')

    eq_lambda = ''
    if 'equivariant' in path_root or 'eq_lmbda' in path_root:
        lambda_match = re.search(r'eq_lmbda_([-\d.e]+)', path_root)
        if lambda_match:
            eq_lambda = str(float(lambda_match.group(1)))
    # For resnet models with invariant_only, set lambda to 0.0 if no eq_lmbda found
    elif 'invariant_only' in full_path_str and arch.startswith('resnet'):
        eq_lambda = '0.0'
    elif 'invariant' in full_path_str and 'kell2018' in arch:
        eq_lambda = '0.0'

    task_parts = []
    if is_ssl:
        if is_dual:
            task_parts.append('ssl dual')
        else:
            task_parts.append('ssl')

        if 'audioset_only' in full_path_str:
            task_parts.append('unbalanced audioset')
        elif 'multi-task' not in modifiers:
            task_parts.extend(modifiers)
    else:
        task_parts.append('supervised')
        task_parts.extend(modifiers)

    if eq_lambda:
        task_parts.append(f'λ={eq_lambda}')

    task = ' '.join(task_parts)

    aug_str = ''
    if 'paired_augmentations' in path_root:
        aug_str = 'paired augmentations'
    elif 'no_shared_fg_augments' in path_root:
        aug_str = 'unpaired augs'

    test_layer = full_path_str.split('task_')[-1].split('_')[0]

    if (arch and arch.lower() == 'byola') or path_str.startswith('config_linear'):
        if 'invariant' in full_path_str:
            model_name = 'byol-a λ=0.0'
        else:
            model_name = 'byol-a'
        return normalize_model_name(model_name), arch, test_layer, eq_lambda

    model_name = f"{arch} {task}"
    if aug_str:
        model_name = f"{model_name} {aug_str}"

    return normalize_model_name(model_name), arch, test_layer, eq_lambda


def format_model_str_nsynth(path):
    """
    Format model string specifically for NSynth results.
    NSynth files have format: model_config_nsynth_task_linear_eval_task_layer_AdamW_0.005.pkl
    
    This function uses format_model_str for model name and architecture extraction,
    but correctly extracts the layer name from the NSynth-specific filename structure.
    """
    path_stem = Path(path).stem
    path_str = str(path)
    
    # Special case: byol-a models with config_nsynth_family_linear_eval_AdamW_final_0.005.pkl
    if 'config_nsynth_family_linear_eval_AdamW_final_0.005.pkl' in path_str:
        model_name = 'byol-a'
        arch_class = 'byol-a'
        eq_lmbda = ''
    else:
        # Use the general format_model_str for model name and architecture
        model_name, arch_class, _, eq_lmbda = format_model_str(path)
    
    # Extract test layer from NSynth-specific filename structure
    # Look for _linear_eval_ and extract layer name after that
    test_layer = 'unknown'
    linear_eval_idx = path_stem.find('_linear_eval_')
    if linear_eval_idx != -1:
        # Get the part after _linear_eval_
        after_linear_eval = path_stem[linear_eval_idx + len('_linear_eval_'):]
        # Split by underscores and find the layer
        parts = after_linear_eval.split('_')
        
        # Common layers to look for
        layer_patterns = [
            'relu0', 'relu1', 'relu2', 'relu3', 'relu4', 'relufc',
            'conv0', 'conv1', 'conv2', 'conv3', 'conv4',
            'maxpool0', 'maxpool1',
            'batchnorm0', 'batchnorm1', 'batchnorm2',
            'avgpool', 'fullyconnected', 'dropout',
            'layer1', 'layer2', 'layer3', 'layer4',
            'bn1', 'conv1_relu1', 'maxpool1',
            'final'
        ]
        
        # Look for layer patterns in the parts
        for part in parts:
            if part in layer_patterns:
                test_layer = part
                break
        
        # If no layer found, try to find it before optimizer (AdamW, SGD, etc.)
        if test_layer == 'unknown':
            optimizer_keywords = ['AdamW', 'SGD', 'LARS']
            for i, part in enumerate(parts):
                if part in optimizer_keywords:
                    # Layer should be the part just before the optimizer
                    if i > 0:
                        test_layer = parts[i-1]
                        break
        
        # If still not found, try the second or third part (after task name)
        if test_layer == 'unknown' and len(parts) > 1:
            # Skip common task names and modifiers
            skip_words = ['family', 'pitch', 'instrument', 'full', 'rep', 'AdamW', 'LARS', 'SGD']
            for part in parts[1:]:
                if part not in skip_words and not part.replace('.', '').isdigit():
                    test_layer = part
                    break
    
    return model_name, arch_class, test_layer, eq_lmbda


def pointplot_by_layer(
    data,
    ax,
    *,
    x='test_layer',
    y='overall_acc',
    hue='model_name',
    order=None,
    hue_order=None,
    palette=None,
    markers=None,
):
    return sns.pointplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        order=order,
        hue_order=hue_order,
        palette=palette,
        ax=ax,
        markers=markers,
        mec='k',
        mew=1,
        lw=1,
        ms=6,
        dodge=0.75,
    )


def pointplot_by_model(
    data,
    ax,
    *,
    x='model_name',
    y='overall_acc',
    order=None,
    hue='model_name',
    palette=None,
):
    return sns.pointplot(
        data=data,
        x=x,
        y=y,
        order=order,
        hue=hue,
        palette=palette,
        ax=ax,
        mec='k',
        mew=1,
        lw=1,
        ms=6,
    )

def plot_sequential_bars(ax, data, value_col, title, ylabel, xlabel='Model', hue_order=None, hue_dict=None, 
                         error_col=None, error_type=None, yerr=None, capsize=4, 
                         bar_width=0.7):
    """
    Create sequential bar plot with models in order.
    
    Args:
        ax: matplotlib axis
        data: DataFrame with model_name and value_col columns
        value_col: Column name for y-axis values
        title: Plot title
        ylabel: Y-axis label
        xlabel: X-axis label
        hue_order: Optional normalized hue_order. If None, uses standard.
        hue_dict: Optional color palette dict. If None, builds from standard colors.
        error_col: Optional column name containing error values. If provided, these will be used.
        error_type: Optional string ('std' or 'sem') to calculate error bars from data.
        yerr: Optional array/Series of error values.
        capsize: Size of error bar caps (default: 4)
        bar_width: Width of each bar (default: 0.7)
    
    Returns:
        Tuple of (ax, legend_handles, legend_labels)
    """
    if hue_order is None:
        hue_order = get_standard_hue_order()
    if hue_dict is None:
        base_colors = get_standard_base_colors()
        hue_dict = build_model_palette(hue_order, base_colors)
    
    means = data.groupby('model_name', as_index=True)[value_col].mean()
    
    # Calculate error bars if requested
    errors = None
    if yerr is not None:
        if hasattr(yerr, 'index'):
            errors = yerr
        else:
            errors = pd.Series(yerr, index=means.index)
    elif error_col is not None:
        errors = data.groupby('model_name', as_index=True)[error_col].mean()
    elif error_type is not None:
        if error_type == 'std':
            errors = data.groupby('model_name', as_index=True)[value_col].std()
        elif error_type == 'sem':
            stds = data.groupby('model_name', as_index=True)[value_col].std()
            counts = data.groupby('model_name', as_index=True)[value_col].count()
            errors = stds / (counts ** 0.5)
        else:
            raise ValueError(f"error_type must be 'std' or 'sem', got '{error_type}'")
    
    # Filter to models present in data
    models = [m for m in hue_order if m in means.index]
    
    # Sequential x positions
    x_positions = np.arange(len(models))
    
    legend_handles = []
    legend_labels = []
    
    for i, model in enumerate(models):
        color = hue_dict.get(model, '#444444')
        label = model_label(model)
        
        # Get error value for this model if errors are provided
        error_val = None
        if errors is not None and model in errors.index:
            error_val = errors.loc[model]
        
        # Build bar arguments
        bar_kwargs = {
            'width': bar_width,
            'color': color,
            'edgecolor': 'black',
            'linewidth': 0.6,
            'label': label,
        }
        if error_val is not None:
            bar_kwargs['yerr'] = error_val
            bar_kwargs['capsize'] = capsize
        
        bar = ax.bar(x_positions[i], means.loc[model], **bar_kwargs)
        legend_handles.append(bar[0])
        legend_labels.append(label)
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels([model_label(m) for m in models], rotation=45, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    
    return ax, legend_handles, legend_labels
    

def plot_grouped_bars(ax, data, value_col, title, ylabel, xlabel='Model Group', hue_order=None, hue_dict=None, 
                      error_col=None, error_type=None, yerr=None, capsize=4, 
                      bar_width=0.18, group_gap=0.6):
    """
    Create grouped bar plot with model groups (Supervised, SSL, Scaled SSL, BYOL-A).
    
    Args:
        ax: matplotlib axis
        data: DataFrame with model_name and value_col columns
        value_col: Column name for y-axis values
        title: Plot title
        ylabel: Y-axis label
        hue_order: Optional normalized hue_order. If None, uses standard.
        hue_dict: Optional color palette dict. If None, builds from standard colors.
        error_col: Optional column name containing error values. If provided, these will be used.
        error_type: Optional string ('std' or 'sem') to calculate error bars from data.
                   If 'std', uses standard deviation. If 'sem', uses standard error of the mean.
        yerr: Optional array/Series of error values. If provided, these will be used directly.
              Should be indexed by model_name to match means.
        capsize: Size of error bar caps (default: 4)
    
    Returns:
        Tuple of (legend_handles, legend_labels)
    """
    if hue_order is None:
        hue_order = get_standard_hue_order()
    if hue_dict is None:
        base_colors = get_standard_base_colors()
        hue_dict = build_model_palette(hue_order, base_colors)
    
    means = (data.groupby('model_name', as_index=True)[value_col].mean())
    
    # Calculate error bars if requested
    errors = None
    if yerr is not None:
        # Use provided error values directly
        if hasattr(yerr, 'index'):
            errors = yerr
        else:
            # Convert to Series with same index as means
            errors = pd.Series(yerr, index=means.index)
    elif error_col is not None:
        # Use error values from specified column
        errors = (data.groupby('model_name', as_index=True)[error_col].mean())
    elif error_type is not None:
        # Calculate error from data
        if error_type == 'std':
            errors = (data.groupby('model_name', as_index=True)[value_col].std())
        elif error_type == 'sem':
            # Standard error of the mean
            stds = (data.groupby('model_name', as_index=True)[value_col].std())
            counts = (data.groupby('model_name', as_index=True)[value_col].count())
            errors = stds / (counts ** 0.5)
        else:
            raise ValueError(f"error_type must be 'std' or 'sem', got '{error_type}'")
    
    plot_groups = get_plot_groups(hue_order)
    
    bar_width = bar_width
    group_gap = group_gap
    
    xticks = []
    xticklabels = []
    legend_handles = []
    legend_labels = []
    
    x_cursor = 0.0
    for group_name, models in plot_groups:
        models = [m for m in models if m in means.index]
        if len(models) == 0:
            continue
        
        group_start = x_cursor
        for i, model in enumerate(models):
            x = x_cursor + i * bar_width
            color = hue_dict.get(model, '#444444')
            label = model_label(model)
            
            # Get error value for this model if errors are provided
            error_val = None
            if errors is not None and model in errors.index:
                error_val = errors.loc[model]
            
            # Build bar arguments, only include yerr and capsize if error_val is provided
            bar_kwargs = {
                'width': bar_width,
                'color': color,
                'edgecolor': 'black',
                'linewidth': 0.6,
                'label': label,
            }
            if error_val is not None:
                bar_kwargs['yerr'] = error_val
                bar_kwargs['capsize'] = capsize
            
            bar = ax.bar(x, means.loc[model], **bar_kwargs)
            legend_handles.append(bar[0])
            legend_labels.append(label)
        
        group_center = group_start + (len(models) - 1) * bar_width / 2
        xticks.append(group_center)
        xticklabels.append(group_name)
        x_cursor += len(models) * bar_width + group_gap
    
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    
    return ax, legend_handles, legend_labels


def plot_fmri_components(results_df, exclude_pattern="spectemp|dual|1.0", 
                         n_cols=6, figsize=None, show_spectemp_line=True,
                         ylim_offset=0.05, average_over_components=False,
                         model_order=None):
    """
    Plot bar charts for fMRI component predictions across models.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with columns: 'model_name_str', 'comp', 'median_r2_test', 
        'median_r2_test_std_over_it'
    exclude_pattern : str, optional
        Regex pattern for model names to exclude from bars (default: "spectemp|dual|1.0")
    n_cols : int, optional
        Number of columns in subplot grid (default: 6)
    figsize : tuple, optional
        Figure size as (width, height). If None, auto-calculated based on grid.
    show_spectemp_line : bool, optional
        Whether to show spectrotemporal reference line (default: True)
    ylim_offset : float, optional
        Offset below spectemp line for y-axis lower limit (default: 0.05)
    average_over_components : bool, optional
        If True, plot a single panel showing average across all components (default: False)
    model_order : list, optional
        Custom list of model names to plot and their order. If None, uses standard order
        plus any additional models found in the data.
    
    Returns
    -------
    fig, axes : matplotlib figure and axes (axes is a single Axes if average_over_components=True)
    """
    # Filter data for plotting
    to_plot = results_df[~results_df.model_name_str.str.contains(exclude_pattern)]
    
    # Build model order - use custom if provided, otherwise standard + any extra models in data
    all_models_in_data = to_plot['model_name_str'].unique()
    if model_order is not None:
        # Use custom order, filter to models actually in data
        model_order = [m for m in model_order if m in all_models_in_data]
    else:
        # Start with standard order, filter to models in data
        std_order = get_standard_hue_order()
        model_order = [m for m in std_order if m in all_models_in_data]
        # Add any models in data that aren't in standard order
        extra_models = [m for m in all_models_in_data if m not in model_order]
        model_order = model_order + extra_models
    
    # Build palette
    base_colors = get_standard_base_colors()
    hue_dict = build_model_palette(model_order, base_colors)
    hue_dict['byol-a'] = "k"
    # Add palette for whisper models from Oranges palette
    whisper_models = [m for m in model_order if m.lower().startswith("whisper")]
    if whisper_models:
        whisper_colors = sns.color_palette('Oranges', n_colors=max(len(whisper_models), 2))
        for i, name in enumerate(whisper_models):
            hue_dict[name] = whisper_colors[i % len(whisper_colors)]
    # Get spectrotemporal values for reference line
    spectemp_vals = results_df[results_df.model_name_str.str.contains("spectemp")]
    
    # Helper function to create x-axis labels
    def get_x_labels(hue_order):
        x_labels = []
        for name in hue_order:
            label = model_label(name)
            if 'scaled ssl' in name:
                if label.startswith('λ='):
                    lambda_val = label.replace('λ=', '')
                    x_labels.append(f"scaled ssl λ={lambda_val}")
                else:
                    x_labels.append(f"scaled ssl {label}")
            elif 'ssl' in name and 'scaled' not in name:
                if label.startswith('λ='):
                    lambda_val = label.replace('λ=', '')
                    x_labels.append(f"ssl λ={lambda_val}")
                else:
                    x_labels.append(f"ssl {label}")
            else:
                x_labels.append(label)
        return x_labels
    
    # Average over components mode - single panel
    if average_over_components:
        if figsize is None:
            figsize = (4, 4)
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Calculate average across components for each model
        avg_data = to_plot.groupby('model_name_str').agg({
            'median_r2_test': 'mean',
            'median_r2_test_std_over_it': 'mean'
        }).reset_index()
        
        # Remove duplicates
        avg_data = avg_data.drop_duplicates(subset=['model_name_str'], keep='first')
        
        # Reorder by model order
        avg_data = avg_data.set_index('model_name_str').loc[model_order].reset_index()
        
        # Get the sorted order
        avg_hue_order = avg_data['model_name_str'].values
        
        # Create x positions
        x_pos = np.arange(len(avg_hue_order))
        
        # Get colors for each model
        colors = [hue_dict.get(name, '#444444') for name in avg_hue_order]
        
        # Create barplot
        bar_width = 0.8
        ax.bar(x_pos, avg_data['median_r2_test'].values, width=bar_width, 
               color=colors, edgecolor='black', linewidth=0.5, zorder=0)
        
        # Add error bars
        errors = avg_data['median_r2_test_std_over_it'].values
        ax.errorbar(x_pos, avg_data['median_r2_test'].values, yerr=errors, 
                    fmt='none', color='black', capsize=0, capthick=1, elinewidth=1, zorder=4)
        
        # Add spectrotemporal reference line (average across components)
        if show_spectemp_line and len(spectemp_vals) > 0:
            spectemp_avg = spectemp_vals.groupby('comp')['median_r2_test'].first().mean()
            ax.axhline(spectemp_avg, color='k', linestyle='--', linewidth=1.5, zorder=5, label='spectemp')
            ax.set_ylim(spectemp_avg - ylim_offset, 1)
        
        # Set labels and title
        ax.set_title('Average over components')
        ax.set_ylabel('mean over median $R^2$')
        ax.set_xlabel('')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(get_x_labels(avg_hue_order), rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
        
        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        if any('spectemp' in label for label in labels):
            ax.legend(loc='upper right', frameon=False)
        
        plt.tight_layout()
        return fig, ax
    
    # Per-component mode - multiple subplots
    # Get unique components
    components = to_plot['comp'].unique()
    
    # Create figure with subplots for each component
    n_components = len(components)
    n_rows = (n_components + n_cols - 1) // n_cols
    
    if figsize is None:
        figsize = (20, 3 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_components == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, component in enumerate(components):
        ax = axes[idx]
        
        # Filter data for this component
        comp_data = to_plot[to_plot['comp'] == component].copy()
        
        # Remove duplicates - keep first occurrence of each model_name_str
        comp_data = comp_data.drop_duplicates(subset=['model_name_str'], keep='first')
        
        # Use model order for this component
        comp_data = comp_data.set_index('model_name_str').loc[model_order].reset_index()
        
        # Get the sorted order for this component
        comp_hue_order = comp_data['model_name_str'].values
        
        # Create x positions
        x_pos = np.arange(len(comp_hue_order))
        
        # Get colors for each model
        colors = [hue_dict.get(name, '#444444') for name in comp_hue_order]
        
        # Create barplot with styling matching plot_grouped_bars
        bar_width = 0.8
        ax.bar(x_pos, comp_data['median_r2_test'].values, width=bar_width, 
               color=colors, edgecolor='black', linewidth=0.5, zorder=0)
        
        # Add error bars (vertical lines) using median_r2_test_std_over_it
        errors = comp_data['median_r2_test_std_over_it'].values
        ax.errorbar(x_pos, comp_data['median_r2_test'].values, yerr=errors, 
                    fmt='none', color='black', capsize=0, capthick=1, elinewidth=1, zorder=4)
        
        # Add spectrotemporal reference line
        if show_spectemp_line and len(spectemp_vals) > 0:
            spectemp_row = spectemp_vals[spectemp_vals.comp == component]
            if len(spectemp_row) > 0:
                spectemp_med = spectemp_row['median_r2_test'].values[0]
                ax.axhline(spectemp_med, color='k', linestyle='--', linewidth=1.5, zorder=5, label='spectemp')
                ax.set_ylim(spectemp_med - ylim_offset, 1)
        
        # Set labels and title
        ax.set_title(f'{component}')
        if idx == 0 or idx == n_cols:
            ax.set_ylabel('median $R^2$')
        ax.set_xlabel('')
        ax.set_xticks(x_pos)
        
        # Create x-axis labels using helper function
        ax.set_xticklabels(get_x_labels(comp_hue_order), rotation=45, ha='right')
        
        ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
    
    # Hide extra subplots if any
    for idx in range(n_components, len(axes)):
        axes[idx].set_visible(False)
    
    # Add legend for spectemp dashed line
    handles, labels = axes[0].get_legend_handles_labels()
    if any('spectemp' in label or 'Spectrotemporal' in label for label in labels):
        axes[0].legend(loc='upper right', frameon=False)
    
    plt.tight_layout()
    return fig, axes


def add_supervised_reference_lines(
    ax,
    df,
    value_col="overall_acc",
    max_x_pos=0,
    min_x_pos=0,
    ylim_pad=(0.95, 1.05),
    fontsize=8,
):
    """
    Add dashed reference lines for best/worst supervised models with arrow annotations.

    Parameters
    ----------
    ax : matplotlib axes
        The axes to draw on
    df : pandas DataFrame
        DataFrame containing all models (will be filtered to supervised only)
    value_col : str
        Column name containing the metric values
    max_x_pos : float
        X position for the max annotation
    min_x_pos : float
        X position for the min annotation
    ylim_pad : tuple
        Padding (bottom, top) to add to ylim beyond the min/max values
    fontsize : int
        Font size for labels
    """
    sup_data = df[df.model_name.str.contains("sup", na=False)]

    if sup_data.empty:
        return None, None

    min_val, max_val = sup_data[value_col].agg(["min", "max"])

    ax.axhline(
        min_val,
        color="black",
        linestyle="--",
        alpha=0.5,
    )
    ax.axhline(
        max_val,
        color="black",
        linestyle="--",
        alpha=0.5,
    )

    ax.set_ylim(min_val * ylim_pad[0], max_val * ylim_pad[1])

    arrow_props = dict(arrowstyle="->", color="black", lw=1)

    ax.annotate(
        "Best\nsupervised",
        xy=(max_x_pos, max_val),
        xytext=(max_x_pos, max_val + 0.04),
        ha="center",
        va="bottom",
        fontsize=fontsize,
        arrowprops=arrow_props,
    )

    ax.annotate(
        "Worst\nsupervised",
        xy=(min_x_pos, min_val),
        xytext=(min_x_pos, max_val + 0.04),
        ha="center",
        va="bottom",
        fontsize=fontsize,
        arrowprops=arrow_props,
    )

    return min_val, max_val


def simple_bar_plot(ax, data, value_col, error_col, title, ylabel, hue_dict, bar_width=1.0, x_pad=0.5, fontsize=12):
    """
    Simple bar plot with lambda values on x-axis.
    
    Parameters
    ----------
    ax : matplotlib axes
        The axes to draw on
    data : pandas DataFrame
        DataFrame containing the data to plot
    value_col : str
        Column name containing the metric values
    error_col : str
        Column name containing error values
    title : str
        Plot title
    ylabel : str
        Y-axis label
    hue_dict : dict
        Dictionary mapping model names to colors
    bar_width : float
        Width of bars
    x_pad : float
        Padding on x-axis
    """
    # Group by model and get means
    means = data.groupby('model_name')[value_col].mean()
    errors = data.groupby('model_name')[error_col].mean() if error_col else None

    # Sort model order by lambda value so bars/labels/colors stay aligned.
    # Prefer lambda from model_name (canonical after normalization), then eq_lmbda.
    model_meta = (
        data.groupby('model_name', as_index=False)['eq_lmbda']
        .first()
        .rename(columns={'eq_lmbda': 'eq_lmbda_first'})
    )

    def _lambda_from_row(row):
        name_match = re.search(r'λ=([-\d.]+)', row['model_name'])
        if name_match:
            try:
                return float(name_match.group(1))
            except ValueError:
                pass
        eq_val = row['eq_lmbda_first']
        if pd.notna(eq_val) and str(eq_val) != '':
            try:
                return float(eq_val)
            except ValueError:
                return np.inf
        return np.inf

    model_meta['lambda_sort'] = model_meta.apply(_lambda_from_row, axis=1)
    model_meta = model_meta.sort_values(['lambda_sort', 'model_name'])
    models = model_meta['model_name'].tolist()

    # Create x positions
    x = np.arange(len(models))

    # Get colors and labels (just the number, no "λ=")
    colors = [hue_dict.get(model, '#444444') for model in models]
    labels = [re.search(r'λ=([0-9.]+)', m).group(1) if 'λ=' in m else m for m in models]

    # Reindex means/errors to match model order
    ordered_means = means.reindex(models)
    ordered_errors = errors.reindex(models) if errors is not None else None

    # Plot bars
    bars = ax.bar(x, ordered_means, width=bar_width, color=colors, edgecolor='black', linewidth=0.5)

    # Add error bars if available
    if ordered_errors is not None:
        ax.errorbar(x, ordered_means, yerr=ordered_errors, fmt='none', color='black', capsize=0, linewidth=1)
    
    # Set labels (no rotation)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel('SSL λ', fontsize=fontsize)
    ax.set_xlim(-x_pad, len(x) + x_pad - 1)
    
    return bars, labels
