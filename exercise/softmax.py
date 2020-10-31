from fastai.callback.fp16 import *
from fastai.vision.all import *
from fastbook import *
import fastbook
fastbook.setup_book()

acts = torch.randn((6, 2)) * 2
acts

plot_function(torch.sigmoid, title='Sigmoid', min=-4, max=4)

plot_function(torch.softmax, title='Softmax')

acts.sigmoid()

(acts[:, 0] - acts[:, 1]).sigmoid()

sm_acts = torch.softmax(acts, dim=1)
sm_acts
