#!/bin/bash


conda create -n EP python=3.9
conda activate EP

conda install s5cmd -c conda-forge # Need for downloading data
conda install -c conda-forge av2 # Argoverse 2 API

conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
conda install -c conda-forge tensorflow==2.10.0
conda install -c conda-forge tensorflow-probability==0.18.0
conda install tqdm
conda install matplotlib
conda install shapely
conda install hashlib

# Deactivate the following line if you don't use jupyter notebook
pip install --user ipykernel 
python -m ipykernel install --user --name=EP

