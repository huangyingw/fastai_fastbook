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
from fastai.vision.all import *
from fastbook import *
import fastbook
fastbook.setup_book()

# hide

# # Other Computer Vision Problems

# ## Multi-Label Classification

# ### The Data

path = untar_data(URLs.PASCAL_2007)

df = pd.read_csv(path / 'train.csv')
df.head()

# ### Sidebar: Pandas and DataFrames

df.iloc[:, 0]

df.iloc[0, :]
# Trailing :s are always optional (in numpy, pytorch, pandas, etc.),
#   so this is equivalent:
df.iloc[0]

df['fname']

tmp_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
tmp_df

tmp_df['c'] = tmp_df['a'] + tmp_df['b']
tmp_df

# ### End sidebar

# ### Constructing a DataBlock

dblock = DataBlock()

dsets = dblock.datasets(df)

len(dsets.train), len(dsets.valid)

x, y = dsets.train[0]
x, y

x['fname']

dblock = DataBlock(get_x=lambda r: r['fname'], get_y=lambda r: r['labels'])
dsets = dblock.datasets(df)
dsets.train[0]


def get_x(r): return r['fname']
def get_y(r): return r['labels']
dblock = DataBlock(get_x=get_x, get_y=get_y)
dsets = dblock.datasets(df)
dsets.train[0]


def get_x(r): return path / 'train' / r['fname']
def get_y(r): return r['labels'].split(' ')
dblock = DataBlock(get_x=get_x, get_y=get_y)
dsets = dblock.datasets(df)
dsets.train[0]

dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                   get_x=get_x, get_y=get_y)
dsets = dblock.datasets(df)
dsets.train[0]

idxs = torch.where(dsets.train[0][1] == 1.)[0]
dsets.train.vocab[idxs]


# +
def splitter(df):
    train = df.index[~df['is_valid']].tolist()
    valid = df.index[df['is_valid']].tolist()
    return train, valid

dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                   splitter=splitter,
                   get_x=get_x,
                   get_y=get_y)

dsets = dblock.datasets(df)
dsets.train[0]
# -

dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                   splitter=splitter,
                   get_x=get_x,
                   get_y=get_y,
                   item_tfms=RandomResizedCrop(128, min_scale=0.35))
dls = dblock.dataloaders(df)

dls.show_batch(nrows=1, ncols=3)

# ### Binary Cross-Entropy

learn = cnn_learner(dls, resnet18)

x, y = to_cpu(dls.train.one_batch())
activs = learn.model(x)
activs.shape

activs[0]


def binary_cross_entropy(inputs, targets):
    inputs = inputs.sigmoid()
    return -torch.where(targets == 1, inputs, 1 - inputs).log().mean()


loss_func = nn.BCEWithLogitsLoss()
loss = loss_func(activs, y)
loss


def say_hello(name, say_what="Hello"): return f"{say_what} {name}."
say_hello('Jeremy'), say_hello('Jeremy', 'Ahoy!')

f = partial(say_hello, say_what="Bonjour")
f("Jeremy"), f("Sylvain")

learn = cnn_learner(dls, resnet50, metrics=partial(accuracy_multi, thresh=0.2))
learn.fine_tune(3, base_lr=3e-3, freeze_epochs=4)

learn.metrics = partial(accuracy_multi, thresh=0.1)
learn.validate()

learn.metrics = partial(accuracy_multi, thresh=0.99)
learn.validate()

preds, targs = learn.get_preds()

accuracy_multi(preds, targs, thresh=0.9, sigmoid=False)

xs = torch.linspace(0.05, 0.95, 29)
accs = [accuracy_multi(preds, targs, thresh=i, sigmoid=False) for i in xs]
plt.plot(xs, accs)

# ## Regression

# ### Assemble the Data

path = untar_data(URLs.BIWI_HEAD_POSE)

# hide
Path.BASE_PATH = path

path.ls().sorted()

(path / '01').ls().sorted()

img_files = get_image_files(path)
def img2pose(x): return Path(f'{str(x)[:-7]}pose.txt')
img2pose(img_files[0])

im = PILImage.create(img_files[0])
im.shape

im.to_thumb(160)

cal = np.genfromtxt(path / '01' / 'rgb.cal', skip_footer=6)
def get_ctr(f):
    ctr = np.genfromtxt(img2pose(f), skip_header=3)
    c1 = ctr[0] * cal[0][0] / ctr[2] + cal[0][2]
    c2 = ctr[1] * cal[1][1] / ctr[2] + cal[1][2]
    return tensor([c1, c2])


get_ctr(img_files[0])

biwi = DataBlock(
    blocks=(ImageBlock, PointBlock),
    get_items=get_image_files,
    get_y=get_ctr,
    splitter=FuncSplitter(lambda o: o.parent.name == '13'),
    batch_tfms=[*aug_transforms(size=(240, 320)),
                Normalize.from_stats(*imagenet_stats)]
)

dls = biwi.dataloaders(path)
dls.show_batch(max_n=9, figsize=(8, 6))

xb, yb = dls.one_batch()
xb.shape, yb.shape

yb[0]

# ### Training a Model

learn = cnn_learner(dls, resnet18, y_range=(-1, 1))


def sigmoid_range(x, lo, hi): return torch.sigmoid(x) * (hi - lo) + lo


plot_function(partial(sigmoid_range, lo=-1, hi=1), min=-4, max=4)

dls.loss_func

learn.lr_find()

lr = 1e-2
learn.fine_tune(3, lr)

math.sqrt(0.0001)

learn.show_results(ds_idx=1, nrows=3, figsize=(6, 8))

# ## Conclusion

# ## Questionnaire

# 1. How could multi-label classification improve the usability of the bear classifier?
# 1. How do we encode the dependent variable in a multi-label classification problem?
# 1. How do you access the rows and columns of a DataFrame as if it was a matrix?
# 1. How do you get a column by name from a DataFrame?
# 1. What is the difference between a `Dataset` and `DataLoader`?
# 1. What does a `Datasets` object normally contain?
# 1. What does a `DataLoaders` object normally contain?
# 1. What does `lambda` do in Python?
# 1. What are the methods to customize how the independent and dependent variables are created with the data block API?
# 1. Why is softmax not an appropriate output activation function when using a one hot encoded target?
# 1. Why is `nll_loss` not an appropriate loss function when using a one-hot-encoded target?
# 1. What is the difference between `nn.BCELoss` and `nn.BCEWithLogitsLoss`?
# 1. Why can't we use regular accuracy in a multi-label problem?
# 1. When is it okay to tune a hyperparameter on the validation set?
# 1. How is `y_range` implemented in fastai? (See if you can implement it yourself and test it without peeking!)
# 1. What is a regression problem? What loss function should you use for such a problem?
# 1. What do you need to do to make sure the fastai library applies the same data augmentation to your inputs images and your target point coordinates?

# ### Further Research

# 1. Read a tutorial about Pandas DataFrames and experiment with a few methods that look interesting to you. See the book's website for recommended tutorials.
# 1. Retrain the bear classifier using multi-label classification. See if you can make it work effectively with images that don't contain any bears, including showing that information in the web application. Try an image with two different kinds of bears. Check whether the accuracy on the single-label dataset is impacted using multi-label classification.
