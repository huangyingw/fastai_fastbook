from fastai.vision.all import *
from fastbook import *
import fastbook
fastbook.setup_book()

path = untar_data(URLs.PASCAL_2007)

df = pd.read_csv(path / 'train.csv')
df.head()


def get_x(r): return path / 'train' / r['fname']
def get_y(r): return r['labels'].split(' ')


def splitter(df):
    train = df.index[~df['is_valid']].tolist()
    valid = df.index[df['is_valid']].tolist()
    return train, valid


dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                   splitter=splitter,
                   get_x=get_x,
                   get_y=get_y,
                   item_tfms=RandomResizedCrop(128, min_scale=0.35))
dls = dblock.dataloaders(df)

learn = cnn_learner(dls, resnet18)

x, y = to_cpu(dls.train.one_batch())
activs = learn.model(x)

loss_func = nn.BCEWithLogitsLoss()
loss = loss_func(activs, y)
loss
