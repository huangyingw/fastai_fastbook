from fastai.vision.all import *
from fastbook import *
import fastbook
fastbook.setup_book()


def binary_cross_entropy(inputs, targets):
    inputs = inputs.sigmoid()
    return -torch.where(targets == 1, inputs, 1 - inputs).log().mean()
