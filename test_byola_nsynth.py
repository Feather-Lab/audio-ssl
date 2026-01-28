#!/usr/bin/env python3
"""Test script to verify BYOL-A NSynth evaluation works."""

import torch
import yaml
import pathlib
import sys
import os

# Add project to path
project_root = os.path.abspath('.')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'lightning_scripts'))

from nsynth_linear_eval_module import NSynthLinearEvalModule

def test_byola_nsynth():
    """Test that BYOL-A NSynth evaluation module initializes and runs."""
    print("Testing BYOL-A NSynth evaluation...")
    
    # Load BYOL-A config
    config_path = pathlib.Path('byol-a/config.yaml')
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    print(f"✓ Loaded config from {config_path}")
    
    # Set up config for NSynth evaluation (same as eval script does)
    config['model'] = {}
    config['hparas'] = {}
    config['model']['arch_kwargs'] = {}
    config['num_workers'] = 4
    config['num_gpus'] = 1
    config['hparas']['batch_size'] = 4
    config['hparas']['global_batch_size'] = 4
    config['hparas']['optimizer'] = 'AdamW'
    config['hparas']['lr'] = 0.005
    config['hparas']['epochs'] = 1
    config['hparas']['lr_schedule'] = False
    config['model']['arch_kwargs']['time_average'] = True
    config['model']['arch_kwargs']['num_classes'] = 11  # NSynth family task
    
    print("✓ Configured for NSynth evaluation")
    
    # Initialize module with BYOL-A
    try:
        module = NSynthLinearEvalModule(
            config=config,
            ckpt_path=None,  # No checkpoint needed for BYOL-A
            layer_out='final',
            num_classes=11,
            supervised_backbone=False,
            w_mlp=False,
            mlp_dim=512,
            with_dropout=False,
            nsynth_root="/mnt/home/igriffith/ceph/datasets/nsynth",
            task='family',
            label_field=None,
            duration=2.0,
            fade_window_duration=0.01,
            sample_rate=16000,  # BYOL-A uses 16kHz
            use_whisper=False,
        )
        print("✓ NSynthLinearEvalModule initialized with BYOL-A")
    except Exception as e:
        print(f"✗ Failed to initialize module: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check that BYOL-A module is initialized
    if not hasattr(module, 'byola_module'):
        print("✗ BYOL-A module not found")
        return False
    
    if module.byola_module is None:
        print("✗ BYOL-A module is None")
        return False
    
    print("✓ BYOL-A module is initialized")
    
    # Test forward pass with dummy data
    try:
        # Create dummy audio: (batch, channels, time) at 16kHz, 1 second
        # BYOL-A expects (batch, time) or (batch, 1, time) - mono audio, 1 second
        batch_size = 2
        dummy_audio = torch.randn(batch_size, 1, 16000)  # 1 second at 16kHz
        
        print(f"✓ Created dummy audio: {dummy_audio.shape}")
        
        # Test BYOL-A module directly first with single sample
        print("Testing BYOL-A module directly with single sample...")
        with torch.no_grad():
            # Test with single sample first
            single_audio = dummy_audio[0:1]  # (1, 1, 32000)
            if single_audio.dim() == 3:
                byola_input = single_audio.squeeze(1)  # (1, 32000)
            else:
                byola_input = single_audio
            print(f"  BYOL-A input shape: {byola_input.shape}")
            byola_output = module.byola_module(byola_input)
            print(f"  BYOL-A output shape: {byola_output.shape}")
            
            # Now test with batch
            print("Testing BYOL-A module with batch...")
            if dummy_audio.dim() == 3:
                byola_input_batch = dummy_audio.squeeze(1)  # (batch, time)
            else:
                byola_input_batch = dummy_audio
            print(f"  BYOL-A batch input shape: {byola_input_batch.shape}")
            byola_output_batch = module.byola_module(byola_input_batch)
            print(f"  BYOL-A batch output shape: {byola_output_batch.shape}")
        
        # Forward pass through full module
        module.eval()
        with torch.no_grad():
            logits = module(dummy_audio)
        
        print(f"✓ Forward pass successful: logits shape = {logits.shape}")
        
        # Check output shape
        expected_shape = (batch_size, 11)  # (batch, num_classes)
        if logits.shape != expected_shape:
            print(f"✗ Wrong output shape: expected {expected_shape}, got {logits.shape}")
            return False
        
        print(f"✓ Output shape correct: {logits.shape}")
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✅ All tests passed! BYOL-A NSynth evaluation is working correctly.")
    return True

if __name__ == '__main__':
    success = test_byola_nsynth()
    sys.exit(0 if success else 1)
