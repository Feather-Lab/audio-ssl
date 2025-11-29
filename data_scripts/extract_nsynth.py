#!/usr/bin/env python3
"""
Utility script to extract NSynth dataset tar.gz files.

The NSynth dataset should be located at /mnt/home/igriffith/ceph/datasets/nsynth
and contain three tar.gz files:
- nsynth-train.jsonwav.tar.gz
- nsynth-valid.jsonwav.tar.gz
- nsynth-test.jsonwav.tar.gz

This script extracts them if not already extracted and verifies the directory structure.
"""

import tarfile
import os
import pathlib
from pathlib import Path


def extract_nsynth(nsynth_root="/mnt/home/igriffith/ceph/datasets/nsynth", force=False):
    """
    Extract NSynth tar.gz files if not already extracted.
    
    Args:
        nsynth_root: Root directory containing NSynth tar.gz files
        force: If True, re-extract even if directories already exist
    
    Returns:
        True if extraction successful or already extracted, False otherwise
    """
    nsynth_root = Path(nsynth_root)
    
    if not nsynth_root.exists():
        raise ValueError(f"NSynth root directory does not exist: {nsynth_root}")
    
    splits = ['train', 'valid', 'test']
    extracted_dirs = []
    
    for split in splits:
        tar_file = nsynth_root / f"nsynth-{split}.jsonwav.tar.gz"
        extract_dir = nsynth_root / f"nsynth-{split}"
        
        if not tar_file.exists():
            print(f"Warning: {tar_file} not found, skipping {split} split")
            continue
        
        # Check if already extracted
        if extract_dir.exists() and extract_dir.is_dir():
            # Verify structure
            audio_dir = extract_dir / "audio"
            examples_json = extract_dir / "examples.json"
            
            if audio_dir.exists() and examples_json.exists():
                if not force:
                    print(f"✓ {split} split already extracted at {extract_dir}")
                    extracted_dirs.append(extract_dir)
                    continue
                else:
                    print(f"Force re-extracting {split} split...")
        
        # Extract
        print(f"Extracting {split} split from {tar_file}...")
        try:
            with tarfile.open(tar_file, 'r:gz') as tar:
                tar.extractall(path=nsynth_root)
            print(f"✓ Extracted {split} split to {extract_dir}")
        except Exception as e:
            print(f"✗ Error extracting {split} split: {e}")
            return False
        
        # Verify structure after extraction
        audio_dir = extract_dir / "audio"
        examples_json = extract_dir / "examples.json"
        
        if not audio_dir.exists():
            print(f"✗ Warning: audio directory not found at {audio_dir}")
        if not examples_json.exists():
            print(f"✗ Warning: examples.json not found at {examples_json}")
        
        extracted_dirs.append(extract_dir)
    
    # Final verification
    print("\nVerifying extracted structure...")
    all_valid = True
    for extract_dir in extracted_dirs:
        audio_dir = extract_dir / "audio"
        examples_json = extract_dir / "examples.json"
        
        if audio_dir.exists() and examples_json.exists():
            # Count audio files
            audio_files = list(audio_dir.glob("*.wav"))
            print(f"✓ {extract_dir.name}: {len(audio_files)} audio files, examples.json found")
        else:
            print(f"✗ {extract_dir.name}: missing required files")
            all_valid = False
    
    return all_valid


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract NSynth dataset tar.gz files")
    parser.add_argument(
        "--nsynth_root",
        type=str,
        default="/mnt/home/igriffith/ceph/datasets/nsynth",
        help="Root directory containing NSynth tar.gz files"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction even if directories already exist"
    )
    
    args = parser.parse_args()
    
    success = extract_nsynth(nsynth_root=args.nsynth_root, force=args.force)
    
    if success:
        print("\n✓ NSynth extraction completed successfully")
        exit(0)
    else:
        print("\n✗ NSynth extraction had issues")
        exit(1)

