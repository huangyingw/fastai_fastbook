# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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

# hide
from fastai.callback.fp16 import *
from fastai.vision.all import *
from fastbook import *
import fastbook
fastbook.setup_book()

# hide

# # Image Classification

# ## From Dogs and Cats to Pet Breeds

path = untar_data(URLs.PETS)

# hide
Path.BASE_PATH = path

path.ls()

(path / "images").ls()

fname = (path / "images").ls()[0]

re.findall(r'(.+)_\d+.jpg$', fname.name)

pets = DataBlock(blocks=(ImageBlock, CategoryBlock),
                 get_items=get_image_files,
                 splitter=RandomSplitter(seed=42),
                 get_y=using_attr(RegexLabeller(r'(.+)_\d+.jpg$'), 'name'),
                 item_tfms=Resize(460),
                 batch_tfms=aug_transforms(size=224, min_scale=0.75))
dls = pets.dataloaders(path / "images")


dls.show_batch(nrows=1, ncols=3)


# ### Softmax

plot_function(torch.sigmoid, min=-4, max=4)

# hide
torch.random.manual_seed(42)

acts = torch.randn((6, 2)) * 2
acts

acts.sigmoid()

(acts[:, 0] - acts[:, 1]).sigmoid()


# ### Log Likelihood

targ = tensor([0, 1, 0, 1, 1, 0])


# ### Taking the Log

plot_function(torch.log, min=0, max=4)


learn = cnn_learner(dls, resnet34, metrics=error_rate)
lr_min, lr_steep = learn.lr_find()

print(f"Minimum/10: {lr_min:.2e}, steepest point: {lr_steep:.2e}")
