import re
from pathlib import Path

import pandas as pd
import seaborn as sns


# Standard model ordering and colors for consistent plotting across notebooks
STANDARD_HUE_ORDER = [
    # Supervised
    'CochDNN9 supervised multi-task',
    'CochDNN9 supervised word',
    'CochDNN9 supervised audioset',
    'CochDNN9 scaled supervised',
    # SSL / Equivariant
    'CochDNN9 ssl λ=0.0',
    'CochDNN9 ssl λ=0.1',
    'CochDNN9 ssl λ=0.2',
    'CochDNN9 ssl λ=0.3',
    'CochDNN9 ssl λ=0.4',
    'CochDNN9 ssl λ=0.5',
    'CochDNN9 scaled ssl λ=0.0',
    'CochDNN9 scaled ssl λ=0.5',
    # SSL / Dual
    'CochDNN9 ssl dual λ=0.0',
    'CochDNN9 ssl dual λ=0.0 unpaired augs',
    'CochDNN9 ssl dual λ=0.01',
    'CochDNN9 ssl dual λ=0.1',
    'CochDNN9 ssl dual λ=0.5',
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
    'CochDNN9 supervised word': 'blue',
    'CochDNN9 supervised audioset': 'orange',
    'CochDNN9 scaled supervised': 'red',
    'CochDNN9 supervised multi-task': 'grey',
}


def get_standard_hue_order():
    """Get the standard model order for plots, normalized."""
    return normalize_model_list(STANDARD_HUE_ORDER)


def get_standard_base_colors():
    """Get the standard base colors for supervised models, normalized."""
    return normalize_palette_dict(STANDARD_BASE_COLORS)


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
        'CochDNN9 supervised multi-task',
        'CochDNN9 supervised word',
        'CochDNN9 supervised audioset',
        'CochDNN9 scaled supervised',
    ]
    ssl_names = [
        name for name in hue_order
        if name.startswith('CochDNN9 ssl ') and 'dual' not in name and 'scaled' not in name
    ]
    scaled_ssl_names = [
        name for name in hue_order if name.startswith('CochDNN9 scaled ssl ')
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


def strip_cochdnn9(name):
    """Remove 'CochDNN9 ' prefix from model name."""
    return name.replace('CochDNN9 ', '')


def lambda_label(name):
    """Extract lambda value from model name for label."""
    match = re.search(r'(?:λ|lambda)\s*=\s*([0-9.]+)', name, re.IGNORECASE)
    return f"λ={match.group(1)}" if match else strip_cochdnn9(name)


def model_label(name):
    """
    Format model name for plot labels.
    For SSL models, shows just lambda value. Otherwise strips CochDNN9 prefix.
    """
    if name.startswith('CochDNN9 ssl ') or name.startswith('CochDNN9 scaled ssl '):
        return lambda_label(name)
    return strip_cochdnn9(name)


def normalize_model_name(model_name):
    name = model_name
    name = re.sub(r'\bkell2018\b', 'CochDNN9', name, flags=re.IGNORECASE)

    name = re.sub(r'\bscaled ssl\b\s+eq\b', 'scaled ssl', name, flags=re.IGNORECASE)
    name = re.sub(r'\bssl\b\s+eq\b', 'ssl', name, flags=re.IGNORECASE)

    # Apply SSL normalization rules to all architectures (CochDNN9, resnet18, resnet50, etc.)
    # Match any architecture prefix followed by ssl word/audioset
    name = re.sub(r'\b(CochDNN9|resnet\d+)\b\s+ssl\s+word\b', r'\1 ssl λ=0.0', name, flags=re.IGNORECASE)
    name = re.sub(r'\b(CochDNN9|resnet\d+)\b\s+ssl\s+audioset\b', r'\1 ssl λ=1.0', name, flags=re.IGNORECASE)

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
                'scaled supervised',
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


def _palette_for_lambdas(hue_order, prefix, palette_name):
    names = [n for n in hue_order if n.startswith(prefix) and 'λ=' in n]
    if not names:
        return {}
    lambdas = sorted({float(_parse_lambda_value(n)) for n in names if _parse_lambda_value(n) is not None})
    colors = sns.color_palette(palette_name, n_colors=max(len(lambdas), 2))
    color_map = dict(zip(lambdas, colors))
    return {n: color_map[_parse_lambda_value(n)] for n in names if _parse_lambda_value(n) is not None}


def _palette_for_lambdas_matching(hue_order, match_fn, palette_name):
    names = [n for n in hue_order if match_fn(n) and 'λ=' in n]
    if not names:
        return {}
    lambdas = sorted({float(_parse_lambda_value(n)) for n in names if _parse_lambda_value(n) is not None})
    if not lambdas:
        return {}
    colors = sns.color_palette(palette_name, n_colors=max(len(lambdas), 2))
    color_map = dict(zip(lambdas, colors))
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
    # Scaled SSL for CochDNN9 - use Purples
    cochdnn9_scaled_ssl = _palette_for_lambdas_matching(
        hue_order, 
        lambda n: 'scaled ssl' in n and n.startswith('CochDNN9'),
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
    
    # Handle supervised CochDNN9 models that aren't in base_colors
    cochdnn9_supervised = [n for n in hue_order if n.startswith('CochDNN9') and 'supervised' in n and n not in palette]
    if cochdnn9_supervised:
        # Use a different palette for supervised CochDNN9 models not in base_colors
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
    
    # Handle any other CochDNN9 models that don't have lambda (fallback)
    cochdnn9_other = [n for n in hue_order if n.startswith('CochDNN9') and n not in palette]
    if cochdnn9_other:
        colors = sns.color_palette('Greens', n_colors=max(len(cochdnn9_other), 2))
        for i, name in enumerate(cochdnn9_other):
            palette[name] = colors[i % len(colors)]

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
    group_gap = 0.6
    
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
    
    return legend_handles, legend_labels
