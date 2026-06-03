#!/usr/bin/env python

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as readme_file:
    readme = readme_file.read()

requirements = [
    "torch",
    "torchaudio",
    "lightning",
    "torchmetrics",
    "h5py",
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "dill",
    "cox",
    "tables",
    "tqdm",
    "pyyaml",
    "resampy",
    "tensorboardX",
    "chcochleagram @ git+https://github.com/jenellefeather/chcochleagram.git",
]

setup(
    name="cochdnn",
    version="1.0.0",
    description="Contrastive-equivariant self-supervised audio representation models.",
    long_description=readme,
    packages=find_packages(include=["audio_ssl*", "lightning_scripts*", "robustness*"]),
    py_modules=["default_paths", "figure_utils"],
    install_requires=requirements,
    license="MIT",
    keywords="audio, self-supervised learning, neural networks",
)
