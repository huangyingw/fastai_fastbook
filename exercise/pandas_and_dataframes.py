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

from fastai.vision.all import *
from fastbook import *
import fastbook
fastbook.setup_book()


path = untar_data(URLs.PASCAL_2007)

df = pd.read_csv(path / 'train.csv')
df.head()

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

dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                   splitter=splitter,
                   get_x=get_x,
                   get_y=get_y,
                   item_tfms=RandomResizedCrop(128, min_scale=0.35))
dls = dblock.dataloaders(df)

dls.show_batch(nrows=1, ncols=3)

# ### Binary Cross-Entropy

learn = cnn_learner(dls, resnet18)

dls.train.one_batch()

x, y = to_cpu(dls.train.one_batch())
activs = learn.model(x)
activs.shape

activs[0]
