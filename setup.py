#!/usr/bin/env python

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="eeg-super-resolution",
    version="1.0.0",
    author="Bo Dai, Xinyu Mou, Xinyuan Zhang, ShanGen Zhang, Xiaorong Gao",
    author_email="your-email@domain.com",
    description="Enhancing Brain-Computer Interface Performance via Self-Supervised EEG Super-Resolution",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/eeg-super-resolution",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
            "isort>=5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "eeg-pretrain=scripts.run_labram_pretraining:main",
            "eeg-finetune=scripts.run_labram_finetune:main",
            "eeg-test=scripts.test_pretrain_on_pretrain_data:main",
        ],
    },
) 