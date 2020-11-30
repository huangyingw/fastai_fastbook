from fastai.tabular.all import *
from fastai.collab import *
from fastbook import *
import fastbook
fastbook.setup_book()

path = untar_data(URLs.ML_100k)

ratings = pd.read_csv(path / 'u.data', delimiter='\t', header=None,
                      names=['user', 'movie', 'rating', 'timestamp'])
ratings.head()

movies = pd.read_csv(path / 'u.item', delimiter='|', encoding='latin-1',
                     usecols=(0, 1), names=('movie', 'title'), header=None)
movies.head()

ratings = ratings.merge(movies)
ratings.head()


dls = CollabDataLoaders.from_df(ratings, item_name='title', bs=64)
dls.show_batch()

dls.classes

learn = collab_learner(dls, n_factors=50, y_range=(0, 5.5))

learn.fit_one_cycle(5, 5e-3, wd=0.1)

learn.model

movie_bias = learn.model.i_bias.weight.squeeze()
idxs = movie_bias.argsort(descending=True)[:5]
[dls.classes['title'][i] for i in idxs]
