from fastbook import *
from fastai.vision.all import *
import fastbook
fastbook.setup_book()

path = untar_data(URLs.MNIST_SAMPLE)

threes = (path / 'train' / '3').ls().sorted()
sevens = (path / 'train' / '7').ls().sorted()
threes

seven_tensors = [tensor(Image.open(o)) for o in sevens]
three_tensors = [tensor(Image.open(o)) for o in threes]
len(three_tensors), len(seven_tensors)


stacked_sevens = torch.stack(seven_tensors).float() / 255
stacked_threes = torch.stack(three_tensors).float() / 255
stacked_threes.shape

train_x = torch.cat([stacked_threes, stacked_sevens]).view(-1, 28 * 28)

train_y = tensor([1] * len(threes) + [0] * len(sevens)).unsqueeze(1)
train_x.shape, train_y.shape


valid_3_tens = torch.stack([tensor(Image.open(o))
                            for o in (path / 'valid' / '3').ls()])
valid_3_tens = valid_3_tens.float() / 255
valid_7_tens = torch.stack([tensor(Image.open(o))
                            for o in (path / 'valid' / '7').ls()])
valid_7_tens = valid_7_tens.float() / 255
valid_3_tens.shape, valid_7_tens.shape


valid_x = torch.cat([valid_3_tens, valid_7_tens]).view(-1, 28 * 28)
valid_y = tensor([1] * len(valid_3_tens) + [0] * len(valid_7_tens)).unsqueeze(1)
valid_dset = list(zip(valid_x, valid_y))

dset = list(zip(train_x, train_y))
dl = DataLoader(dset, batch_size=256)

linear_model = nn.Linear(28 * 28, 1)
lr = 1.


valid_dl = DataLoader(valid_dset, batch_size=256)


def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds > 0.5) == yb
    return correct.float().mean()


def validate_epoch(model):
    accs = [batch_accuracy(model(xb), yb) for xb, yb in valid_dl]
    return round(torch.stack(accs).mean().item(), 4)


def mnist_loss(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets == 1, 1 - predictions, predictions).mean()


def calc_grad(xb, yb, model):
    preds = model(xb)
    loss = mnist_loss(preds, yb)
    loss.backward()


class BasicOptim:
    def __init__(self, params, lr): self.params, self.lr = list(params), lr

    def step(self, *args, **kwargs):
        for p in self.params:
            p.data -= p.grad.data * self.lr

    def zero_grad(self, *args, **kwargs):
        for p in self.params:
            p.grad = None


opt = BasicOptim(linear_model.parameters(), lr)


def train_epoch(model):
    for xb, yb in dl:
        calc_grad(xb, yb, model)
        opt.step()
        opt.zero_grad()


def train_model(model, epochs):
    for i in range(epochs):
        train_epoch(model)
        print(validate_epoch(model), end=' ')


train_model(linear_model, 20)
