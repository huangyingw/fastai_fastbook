from fastai.tabular.all import *
from fastai.collab import *
from fastbook import *
import fastbook
fastbook.setup_book()


learn = collab_learner(dls, n_factors=50, y_range=(0, 5.5))

learn.fit_one_cycle(5, 5e-3, wd=0.1)

learn.model

movie_bias = learn.model.i_bias.weight.squeeze()
idxs = movie_bias.argsort(descending=True)[:5]
[dls.classes['title'][i] for i in idxs]
