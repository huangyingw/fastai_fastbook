# -*- coding: utf-8 -*-
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
from fastbook import *
from fastai.vision.all import *
import fastbook
fastbook.setup_book()


path = untar_data(URLs.MNIST_SAMPLE)

Path.BASE_PATH = path

threes = (path / 'train' / '3').ls().sorted()
sevens = (path / 'train' / '7').ls().sorted()
threes

im3_path = threes[1]
im3 = Image.open(im3_path)
im3

seven_tensors = [tensor(Image.open(o)) for o in sevens]
three_tensors = [tensor(Image.open(o)) for o in threes]
stacked_sevens = torch.stack(seven_tensors).float() / 255
stacked_threes = torch.stack(three_tensors).float() / 255
train_x = torch.cat([stacked_threes, stacked_sevens]).view(-1, 28 * 28)

valid_3_tens = torch.stack([tensor(Image.open(o))
                            for o in (path / 'valid' / '3').ls()])
valid_3_tens = valid_3_tens.float() / 255
valid_7_tens = torch.stack([tensor(Image.open(o))
                            for o in (path / 'valid' / '7').ls()])
valid_7_tens = valid_7_tens.float() / 255
valid_x = torch.cat([valid_3_tens, valid_7_tens]).view(-1, 28 * 28)
valid_y = tensor([1] * len(valid_3_tens) + [0] * len(valid_7_tens)).unsqueeze(1)
valid_dset = list(zip(valid_x, valid_y))

train_y = tensor([1] * len(threes) + [0] * len(sevens)).unsqueeze(1)
dset = list(zip(train_x, train_y))

dl = DataLoader(dset, batch_size=256)
valid_dl = DataLoader(valid_dset, batch_size=256)
dls = DataLoaders(dl, valid_dl)

lr = 1.

def mnist_loss(predictions, targets):
    return torch.where(targets == 1, 1 - predictions, predictions).mean()

def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds > 0.5) == yb
    return correct.float().mean()

learn = Learner(dls, nn.Linear(28 * 28, 1), opt_func=SGD,
                loss_func=mnist_loss, metrics=batch_accuracy)

learn.fit(10, lr=lr)
