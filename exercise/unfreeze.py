from fastai.callback.fp16 import *
from fastai.vision.all import *
from fastbook import *
import fastbook
fastbook.setup_book()


learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fit_one_cycle(3, 3e-3)

learn.unfreeze()

learn.lr_find()

learn.fit_one_cycle(6, lr_max=1e-5)
