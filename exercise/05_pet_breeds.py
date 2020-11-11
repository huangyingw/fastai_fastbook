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
from IPython.display import HTML
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
                 splitter=RandomSubsetSplitter(train_sz=0.05, valid_sz=0.01, seed=42),
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

pets1 = DataBlock(blocks=(ImageBlock, CategoryBlock),
                  get_items=get_image_files,
                  splitter=RandomSubsetSplitter(train_sz=0.05, valid_sz=0.01, seed=42),
                  get_y=using_attr(RegexLabeller(r'(.+)_\d+.jpg$'), 'name'))
# pets1.summary(path / "images")

learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(2)

# ## Cross-Entropy Loss

# ### Viewing Activations and Labels

x, y = dls.one_batch()

y

preds, _ = learn.get_preds(dl=[(x, y)])
preds[0]

len(preds[0]), preds[0].sum()

# ### Softmax

plot_function(torch.sigmoid, min=-4, max=4)

# hide
torch.random.manual_seed(42)

acts = torch.randn((6, 2)) * 2
acts

acts.sigmoid()

(acts[:, 0] - acts[:, 1]).sigmoid()

sm_acts = torch.softmax(acts, dim=1)
sm_acts

# ### Log Likelihood

targ = tensor([0, 1, 0, 1, 1, 0])

sm_acts

idx = range(6)
sm_acts[idx, targ]

df = pd.DataFrame(sm_acts, columns=["3", "7"])
df['targ'] = targ
df['idx'] = idx
df['loss'] = sm_acts[range(6), targ]
t = df.style.hide_index()
# To have html code compatible with our script
html = t._repr_html_().split('</style>')[1]
html = re.sub(r'<table id="([^"]+)"\s*>', r'<table >', html)
display(HTML(html))

-sm_acts[idx, targ]

F.nll_loss(sm_acts, targ, reduction='none')

# ### Taking the Log

plot_function(torch.log, min=0, max=4)

loss_func = nn.CrossEntropyLoss()

loss_func(acts, targ)

F.cross_entropy(acts, targ)

nn.CrossEntropyLoss(reduction='none')(acts, targ)

# ## Model Interpretation

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(12, 12), dpi=60)

interp.most_confused(min_val=5)

# ## Improving Our Model

# ### The Learning Rate Finder

learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1, base_lr=0.1)

learn = cnn_learner(dls, resnet34, metrics=error_rate)
lr_min, lr_steep = learn.lr_find()

print(f"Minimum/10: {lr_min:.2e}, steepest point: {lr_steep:.2e}")

learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(2, base_lr=3e-3)

# ### Unfreezing and Transfer Learning

learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fit_one_cycle(3, 3e-3)

learn.unfreeze()

learn.lr_find()

learn.fit_one_cycle(6, lr_max=1e-5)

# ### Discriminative Learning Rates

learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fit_one_cycle(3, 3e-3)
learn.unfreeze()
learn.fit_one_cycle(12, lr_max=slice(1e-6, 1e-4))

learn.recorder.plot_loss()

# ### Selecting the Number of Epochs

# ### Deeper Architectures

learn = cnn_learner(dls, resnet50, metrics=error_rate).to_fp16()
learn.fine_tune(6, freeze_epochs=3)

# ## Conclusion

# ## Questionnaire

# 1. Why do we first resize to a large size on the CPU, and then to a smaller size on the GPU?
# 1. If you are not familiar with regular expressions, find a regular expression tutorial, and some problem sets, and complete them. Have a look on the book's website for suggestions.
# 1. What are the two ways in which data is most commonly provided, for most deep learning datasets?
# 1. Look up the documentation for `L` and try using a few of the new methods is that it adds.
# 1. Look up the documentation for the Python `pathlib` module and try using a few methods of the `Path` class.
# 1. Give two examples of ways that image transformations can degrade the quality of the data.
# 1. What method does fastai provide to view the data in a `DataLoaders`?
# 1. What method does fastai provide to help you debug a `DataBlock`?
# 1. Should you hold off on training a model until you have thoroughly cleaned your data?
# 1. What are the two pieces that are combined into cross-entropy loss in PyTorch?
# 1. What are the two properties of activations that softmax ensures? Why is this important?
# 1. When might you want your activations to not have these two properties?
# 1. Calculate the `exp` and `softmax` columns of <<bear_softmax>> yourself (i.e., in a spreadsheet, with a calculator, or in a notebook).
# 1. Why can't we use `torch.where` to create a loss function for datasets where our label can have more than two categories?
# 1. What is the value of log(-2)? Why?
# 1. What are two good rules of thumb for picking a learning rate from the learning rate finder?
# 1. What two steps does the `fine_tune` method do?
# 1. In Jupyter Notebook, how do you get the source code for a method or function?
# 1. What are discriminative learning rates?
# 1. How is a Python `slice` object interpreted when passed as a learning rate to fastai?
# 1. Why is early stopping a poor choice when using 1cycle training?
# 1. What is the difference between `resnet50` and `resnet101`?
# 1. What does `to_fp16` do?

# ### Further Research

# 1. Find the paper by Leslie Smith that introduced the learning rate finder, and read it.
# 1. See if you can improve the accuracy of the classifier in this chapter. What's the best accuracy you can achieve? Look on the forums and the book's website to see what other students have achieved with this dataset, and how they did it.
