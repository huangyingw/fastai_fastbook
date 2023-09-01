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
n_movies = len(dls.classes['title'])
n_factors = 5


class DotProductBias(Module):
    def __init__(self, n_users, n_movies, n_factors, y_range=(0, 5.5)):
        self.user_factors = Embedding(n_users, n_factors)
        self.user_bias = Embedding(n_users, 1)
        self.movie_factors = Embedding(n_movies, n_factors)
        self.movie_bias = Embedding(n_movies, 1)
        self.y_range = y_range

    def forward(self, x):
        users = self.user_factors(x[:, 0])
        movies = self.movie_factors(x[:, 1])
        res = (users * movies).sum(dim=1, keepdim=True)
        res += self.user_bias(x[:, 0]) + self.movie_bias(x[:, 1])
        return sigmoid_range(res, *self.y_range)


x = np.linspace(-2, 2, 100)
a_s = [1, 2, 5, 10, 50]
ys = [a * x**2 for a in a_s]
_, ax = plt.subplots(figsize=(8, 6))
for a, y in zip(a_s, ys):
    ax.plot(x, y, label=f'a={a}')
ax.set_ylim([0, 5])
ax.legend()

model = DotProductBias(n_users, n_movies, 50)
learn = Learner(dls, model, loss_func=MSELossFlat())
learn.fit_one_cycle(5, 5e-3, wd=0.1)
