from fastai.callback.fp16 import *
from fastai.vision.all import *
from fastbook import *
import fastbook
fastbook.setup_book()

acts = torch.randn((6, 2)) * 2
acts

targ = tensor([0, 1, 0, 1, 1, 0])

sm_acts = torch.softmax(acts, dim=1)
sm_acts

idx = range(6)
-sm_acts[idx, targ]

F.nll_loss(sm_acts, targ, reduction='none')
