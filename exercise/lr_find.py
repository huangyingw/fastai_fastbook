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

# ## Presizing

# +
dblock1 = DataBlock(blocks=(ImageBlock(), CategoryBlock()),
                    get_y=parent_label,
                    item_tfms=Resize(460))
dls1 = dblock1.dataloaders([(Path.cwd() / 'images' / 'grizzly.jpg')] * 100, bs=8)
dls1.train.get_idxs = lambda: Inf.ones
x, y = dls1.valid.one_batch()
_, axs = subplots(1, 2)

x1 = TensorImage(x.clone())
x1 = x1.affine_coord(sz=224)
x1 = x1.rotate(draw=30, p=1.)
x1 = x1.zoom(draw=1.2, p=1.)
x1 = x1.warp(draw_x=-0.2, draw_y=0.2, p=1.)

tfms = setup_aug_tfms([Rotate(draw=30, p=1, size=224), Zoom(draw=1.2, p=1., size=224),
                       Warp(draw_x=-0.2, draw_y=0.2, p=1., size=224)])
x = Pipeline(tfms)(x)
#x.affine_coord(coord_tfm=coord_tfm, sz=size, mode=mode, pad_mode=pad_mode)
TensorImage(x[0]).show(ctx=axs[0])
TensorImage(x1[0]).show(ctx=axs[1])
# -

# ### Checking and Debugging a DataBlock

dls.show_batch(nrows=1, ncols=3)


# ## Cross-Entropy Loss

# ### Viewing Activations and Labels

x, y = dls.one_batch()
y

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

loss_func = nn.CrossEntropyLoss()

loss_func(acts, targ)

F.cross_entropy(acts, targ)

nn.CrossEntropyLoss(reduction='none')(acts, targ)


learn = cnn_learner(dls, resnet34, metrics=error_rate)
lr_min, lr_steep = learn.lr_find()

print(f"Minimum/10: {lr_min:.2e}, steepest point: {lr_steep:.2e}")
