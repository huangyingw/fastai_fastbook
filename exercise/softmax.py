from fastai.callback.fp16 import *
from fastai.vision.all import *
from fastbook import *
import fastbook

fastbook.setup_book()
sm_acts = torch.softmax(acts, dim=1)
sm_acts
