from fastai.vision.all import *
from fastbook import *
import fastbook
fastbook.setup_book()


loss_func = nn.BCEWithLogitsLoss()
loss = loss_func(activs, y)
loss
