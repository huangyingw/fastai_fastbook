from fastbook import *
from fastai.vision.all import *
import fastbook
fastbook.setup_book()


def f(x): return x**2


plot_function(f, 'x', 'x**2')

plot_function(f, 'x', 'x**2')
plt.scatter(-1.5, f(-1.5), color='red')


xt = tensor(3.).requires_grad_()

yt = f(xt)
yt

yt.backward()

xt.grad

xt = tensor([3., 4., 10.]).requires_grad_()
xt


def f(x): return (x**2).sum()


yt = f(xt)
yt

yt.backward()
xt.grad
