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


class BasicOptim:
    def __init__(self, params, lr): self.params, self.lr = list(params), lr

    def step(self, *args, **kwargs):
        for p in self.params:
            p.data -= p.grad.data * self.lr

    def zero_grad(self, *args, **kwargs):
        for p in self.params:
            p.grad = None


lr = 1.
opt = SGD(linear_model.parameters(), lr)

dl = DataLoader(dset, batch_size=256)
xb, yb = first(dl)


def mnist_loss(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets == 1, 1 - predictions, predictions).mean()


def calc_grad(xb, yb, model):
    preds = model(xb)
    loss = mnist_loss(preds, yb)
    loss.backward()


def train_epoch(model):
    for xb, yb in dl:
        calc_grad(xb, yb, model)
        opt.step()
        opt.zero_grad()


valid_dl = DataLoader(valid_dset, batch_size=256)


def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds > 0.5) == yb
    return correct.float().mean()


def validate_epoch(model):
    accs = [batch_accuracy(model(xb), yb) for xb, yb in valid_dl]
    return round(torch.stack(accs).mean().item(), 4)


validate_epoch(linear_model)


def train_model(model, epochs):
    for i in range(epochs):
        train_epoch(model)
        print(validate_epoch(model), end=' ')


train_model(linear_model, 20)

linear_model = nn.Linear(28 * 28, 1)
train_model(linear_model, 20)

dls = DataLoaders(dl, valid_dl)

learn = Learner(dls, nn.Linear(28 * 28, 1), opt_func=SGD,
                loss_func=mnist_loss, metrics=batch_accuracy)

learn.fit(10, lr=lr)
