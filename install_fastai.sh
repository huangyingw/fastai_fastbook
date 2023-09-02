#!/bin/bash -
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cd "$SCRIPTPATH"

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/Users/huangyingw/anaconda3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/Users/huangyingw/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/Users/huangyingw/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/Users/huangyingw/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# Create a new conda environment with Python 3.8
conda create -n fastai python=3.8 -y

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
