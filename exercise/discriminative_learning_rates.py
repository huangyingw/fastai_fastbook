# ---
# jupyter:
#   jupytext:
#     split_at_heading: true
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from fastai.callback.fp16 import *
from fastai.vision.all import *
from fastbook import *
import fastbook
fastbook.setup_book()

path = untar_data(URLs.PETS)
pets = DataBlock(blocks=(ImageBlock, CategoryBlock),
                 get_items=get_image_files,
                 splitter=RandomSubsetSplitter(train_sz=0.3, valid_sz=0.1, seed=42),
                 get_y=using_attr(RegexLabeller(r'(.+)_\d+.jpg$'), 'name'),
                 item_tfms=Resize(460),
                 batch_tfms=aug_transforms(size=224, min_scale=0.75))
dls = pets.dataloaders(path / "images")

learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fit_one_cycle(1, 3e-3)
learn.unfreeze()
learn.fit_one_cycle(3, lr_max=slice(1e-6, 1e-4))

learn.recorder.plot_loss()
