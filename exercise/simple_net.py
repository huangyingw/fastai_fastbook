from fastbook import *
from fastai.vision.all import *
import fastbook
fastbook.setup_book()


path = untar_data(URLs.MNIST_SAMPLE)

threes = (path / 'train' / '3').ls().sorted()
sevens = (path / 'train' / '7').ls().sorted()
seven_tensors = [tensor(Image.open(o)) for o in sevens]
three_tensors = [tensor(Image.open(o)) for o in threes]
stacked_sevens = torch.stack(seven_tensors).float() / 255
stacked_threes = torch.stack(three_tensors).float() / 255

train_x = torch.cat([stacked_threes, stacked_sevens]).view(-1, 28 * 28)

train_y = tensor([1] * len(threes) + [0] * len(sevens)).unsqueeze(1)
train_x.shape, train_y.shape

dset = list(zip(train_x, train_y))

valid_3_tens = torch.stack([tensor(Image.open(o))
                            for o in (path / 'valid' / '3').ls()])
valid_3_tens = valid_3_tens.float() / 255
valid_7_tens = torch.stack([tensor(Image.open(o))
                            for o in (path / 'valid' / '7').ls()])
valid_7_tens = valid_7_tens.float() / 255

valid_x = torch.cat([valid_3_tens, valid_7_tens]).view(-1, 28 * 28)
valid_y = tensor([1] * len(valid_3_tens) + [0] * len(valid_7_tens)).unsqueeze(1)
valid_dset = list(zip(valid_x, valid_y))
# ### Creating an Optimizer

linear_model = nn.Linear(28 * 28, 1)

w, b = linear_model.parameters()
w.shape, b.shape


dl = DataLoader(dset, batch_size=256)


def init_params(size, std=1.0): return (torch.randn(size) * std).requires_grad_()


plot_function(F.relu)

simple_net = nn.Sequential(
    nn.Linear(28 * 28, 30),
    nn.ReLU(),
    nn.Linear(30, 1)
)


def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds > 0.5) == yb
    return correct.float().mean()


def mnist_loss(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets == 1, 1 - predictions, predictions).mean()


valid_dl = DataLoader(valid_dset, batch_size=256)
dls = DataLoaders(dl, valid_dl)
learn = Learner(dls, simple_net, opt_func=SGD,
                loss_func=mnist_loss, metrics=batch_accuracy)

learn.fit(40, 0.1)

plt.plot(L(learn.recorder.values).itemgot(2))

learn.recorder.values[-1][2]
