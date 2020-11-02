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
