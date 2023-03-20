#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="dscript",
    description="D-SCRIPT: protein-protein interaction prediction",
    author="Samuel Sledzieski",
    author_email="samsl@mit.edu",
    url="http://dscript.csail.mit.edu",
    license="GPLv3",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "dscript = dscript.__main__:main",
        ],
    },
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "torch>=1.11",
        "biopython",
        "matplotlib",
        "seaborn",
        "tqdm",
        "scikit-learn",
        "h5py",
    ],
)
