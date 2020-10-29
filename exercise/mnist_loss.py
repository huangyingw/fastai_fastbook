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

from fastbook import *
from fastai.vision.all import *
import fastbook
fastbook.setup_book()

path = untar_data(URLs.MNIST_SAMPLE)
threes = (path / 'train' / '3').ls().sorted()
sevens = (path / 'train' / '7').ls().sorted()
threes

three_tensors = [tensor(Image.open(o)) for o in threes]
seven_tensors = [tensor(Image.open(o)) for o in sevens]
len(three_tensors), len(seven_tensors)

show_image(three_tensors[1])
stacked_threes = torch.stack(three_tensors).float() / 255
stacked_threes.shape
stacked_sevens = torch.stack(seven_tensors).float() / 255

train_x = torch.cat([stacked_threes, stacked_sevens]).view(-1, 28 * 28)
train_y = tensor([1] * len(threes) + [0] * len(sevens)).unsqueeze(1)


def mnist_loss(predictions, targets):
    return torch.where(targets == 1, 1 - predictions, predictions).mean()


def init_params(size, std=1.0): return (torch.randn(size) * std).requires_grad_()


trgts = tensor([1, 0, 1])
prds = tensor([0.9, 0.4, 0.2])

weights = init_params((28 * 28, 1))
bias = init_params(1)

mnist_loss(prds, trgts)

mnist_loss(tensor([0.9, 0.4, 0.8]), trgts)


def mnist_loss(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets == 1, 1 - predictions, predictions).mean()


def linear1(xb): return xb @ weights + bias


batch = train_x[:4]
batch.shape

preds = linear1(batch)
preds

loss = mnist_loss(preds, train_y[:4])
loss
