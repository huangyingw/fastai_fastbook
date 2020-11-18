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

n_users = len(dls.classes['user'])
n_factors = 5
n_users

user_factors = torch.randn(n_users, n_factors)
user_factors

user_factors.t().shape

user_factors.t()

one_hot_3 = one_hot(3, n_users).float()
one_hot_3.shape, one_hot_3[0].shape

one_hot_3[0]

one_hot_3

user_factors.t() @ one_hot_3

user_factors[3]
