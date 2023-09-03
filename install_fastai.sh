#!/bin/bash -
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cd "$SCRIPTPATH"

ANACONDA_PATH="$HOME/anaconda3"

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('$ANACONDA_PATH/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$ANACONDA_PATH/etc/profile.d/conda.sh" ]; then
        . "$ANACONDA_PATH/etc/profile.d/conda.sh"
    else
        export PATH="$ANACONDA_PATH/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# Check if the environment "fastai" already exists
conda info --envs | awk '{print $1}' | grep -w "fastai" &>/dev/null

# If the environment doesn't exist, create it
if [ $? -ne 0 ]; then
    # Create a new conda environment with Python 3.8
    conda create -n fastai python=3.8 -y
fi

# Activate the conda environment
conda activate fastai

# Install PyTorch, torchvision, and torchaudio
conda install pytorch torchvision torchaudio -c pytorch -y

# Install fastai library
pip install fastai

# Install Jupyter Notebook
conda install notebook -y

# Install fastbook
conda install -c fastai fastbook -y
