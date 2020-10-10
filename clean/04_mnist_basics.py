# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     split_at_heading: true
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

#hide
import fastbook
fastbook.setup_book()

# +
#hide
from fastai.vision.all import *
from fastbook import *

matplotlib.rc('image', cmap='Greys')
# -

# # Under the Hood: Training a Digit Classifier

# ## Pixels: The Foundations of Computer Vision

# ## Sidebar: Tenacity and Deep Learning

# ## End sidebar

path = untar_data(URLs.MNIST_SAMPLE)

#hide
Path.BASE_PATH = path

path.ls()

(path/'train').ls()

threes = (path/'train'/'3').ls().sorted()
sevens = (path/'train'/'7').ls().sorted()
threes

im3_path = threes[1]
im3 = Image.open(im3_path)
im3

array(im3)[4:10,4:10]

tensor(im3)[4:10,4:10]

im3_t = tensor(im3)
df = pd.DataFrame(im3_t[4:15,4:22])
df.style.set_properties(**{'font-size':'6pt'}).background_gradient('Greys')

# ## First Try: Pixel Similarity

seven_tensors = [tensor(Image.open(o)) for o in sevens]
three_tensors = [tensor(Image.open(o)) for o in threes]
len(three_tensors),len(seven_tensors)

show_image(three_tensors[1]);

stacked_sevens = torch.stack(seven_tensors).float()/255
stacked_threes = torch.stack(three_tensors).float()/255
stacked_threes.shape

len(stacked_threes.shape)

stacked_threes.ndim

mean3 = stacked_threes.mean(0)
show_image(mean3);

mean7 = stacked_sevens.mean(0)
show_image(mean7);

a_3 = stacked_threes[1]
show_image(a_3);

dist_3_abs = (a_3 - mean3).abs().mean()
dist_3_sqr = ((a_3 - mean3)**2).mean().sqrt()
dist_3_abs,dist_3_sqr

dist_7_abs = (a_3 - mean7).abs().mean()
dist_7_sqr = ((a_3 - mean7)**2).mean().sqrt()
dist_7_abs,dist_7_sqr

F.l1_loss(a_3.float(),mean7), F.mse_loss(a_3,mean7).sqrt()

# ### NumPy Arrays and PyTorch Tensors

data = [[1,2,3],[4,5,6]]
arr = array (data)
tns = tensor(data)

arr  # numpy

tns  # pytorch

tns[1]

tns[:,1]

tns[1,1:3]

tns+1

tns.type()

tns*1.5

# ## Computing Metrics Using Broadcasting

valid_3_tens = torch.stack([tensor(Image.open(o))
                            for o in (path/'valid'/'3').ls()])
valid_3_tens = valid_3_tens.float()/255
valid_7_tens = torch.stack([tensor(Image.open(o))
                            for o in (path/'valid'/'7').ls()])
valid_7_tens = valid_7_tens.float()/255
valid_3_tens.shape,valid_7_tens.shape


def mnist_distance(a,b): return (a-b).abs().mean((-1,-2))
mnist_distance(a_3, mean3)

valid_3_dist = mnist_distance(valid_3_tens, mean3)
valid_3_dist, valid_3_dist.shape

tensor([1,2,3]) + tensor([1,1,1])

(valid_3_tens-mean3).shape


def is_3(x): return mnist_distance(x,mean3) < mnist_distance(x,mean7)


is_3(a_3), is_3(a_3).float()

is_3(valid_3_tens)

# +
accuracy_3s =      is_3(valid_3_tens).float() .mean()
accuracy_7s = (1 - is_3(valid_7_tens).float()).mean()

accuracy_3s,accuracy_7s,(accuracy_3s+accuracy_7s)/2
# -

# ## Stochastic Gradient Descent (SGD)

gv('''
init->predict->loss->gradient->step->stop
step->predict[label=repeat]
''')


def f(x): return x**2


plot_function(f, 'x', 'x**2')

plot_function(f, 'x', 'x**2')
plt.scatter(-1.5, f(-1.5), color='red');

# ### Calculating Gradients

xt = tensor(3.).requires_grad_()

yt = f(xt)
yt

yt.backward()

xt.grad

xt = tensor([3.,4.,10.]).requires_grad_()
xt


# +
def f(x): return (x**2).sum()

yt = f(xt)
yt
# -

yt.backward()
xt.grad

# ### Stepping With a Learning Rate

# ### An End-to-End SGD Example

time = torch.arange(0,20).float(); time

speed = torch.randn(20)*3 + 0.75*(time-9.5)**2 + 1
plt.scatter(time,speed);


def f(t, params):
    a,b,c = params
    return a*(t**2) + (b*t) + c


def mse(preds, targets): return ((preds-targets)**2).mean()


# #### Step 1: Initialize the parameters

params = torch.randn(3).requires_grad_()

#hide
orig_params = params.clone()

# #### Step 2: Calculate the predictions

preds = f(time, params)


def show_preds(preds, ax=None):
    if ax is None: ax=plt.subplots()[1]
    ax.scatter(time, speed)
    ax.scatter(time, to_np(preds), color='red')
    ax.set_ylim(-300,100)


show_preds(preds)

# #### Step 3: Calculate the loss

loss = mse(preds, speed)
loss

# #### Step 4: Calculate the gradients

loss.backward()
params.grad

params.grad * 1e-5

params

# #### Step 5: Step the weights.

lr = 1e-5
params.data -= lr * params.grad.data
params.grad = None

preds = f(time,params)
mse(preds, speed)

show_preds(preds)


def apply_step(params, prn=True):
    preds = f(time, params)
    loss = mse(preds, speed)
    loss.backward()
    params.data -= lr * params.grad.data
    params.grad = None
    if prn: print(loss.item())
    return preds


# #### Step 6: Repeat the process

for i in range(10): apply_step(params)

#hide
params = orig_params.detach().requires_grad_()

_,axs = plt.subplots(1,4,figsize=(12,3))
for ax in axs: show_preds(apply_step(params, False), ax)
plt.tight_layout()

# #### Step 7: stop

# ### Summarizing Gradient Descent

gv('''
init->predict->loss->gradient->step->stop
step->predict[label=repeat]
''')

# ## The MNIST Loss Function

train_x = torch.cat([stacked_threes, stacked_sevens]).view(-1, 28*28)

train_y = tensor([1]*len(threes) + [0]*len(sevens)).unsqueeze(1)
train_x.shape,train_y.shape

dset = list(zip(train_x,train_y))
x,y = dset[0]
x.shape,y

valid_x = torch.cat([valid_3_tens, valid_7_tens]).view(-1, 28*28)
valid_y = tensor([1]*len(valid_3_tens) + [0]*len(valid_7_tens)).unsqueeze(1)
valid_dset = list(zip(valid_x,valid_y))


def init_params(size, std=1.0): return (torch.randn(size)*std).requires_grad_()


weights = init_params((28*28,1))

bias = init_params(1)

(train_x[0]*weights.T).sum() + bias


def linear1(xb): return xb@weights + bias
preds = linear1(train_x)
preds

corrects = (preds>0.0).float() == train_y
corrects

corrects.float().mean().item()

weights[0] *= 1.0001

preds = linear1(train_x)
((preds>0.0).float() == train_y).float().mean().item()

trgts  = tensor([1,0,1])
prds   = tensor([0.9, 0.4, 0.2])


def mnist_loss(predictions, targets):
    return torch.where(targets==1, 1-predictions, predictions).mean()


torch.where(trgts==1, 1-prds, prds)

mnist_loss(prds,trgts)

mnist_loss(tensor([0.9, 0.4, 0.8]),trgts)


# ### Sigmoid

def sigmoid(x): return 1/(1+torch.exp(-x))


plot_function(torch.sigmoid, title='Sigmoid', min=-4, max=4)


def mnist_loss(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets==1, 1-predictions, predictions).mean()


# ### SGD and Mini-Batches

coll = range(15)
dl = DataLoader(coll, batch_size=5, shuffle=True)
list(dl)

ds = L(enumerate(string.ascii_lowercase))
ds

dl = DataLoader(ds, batch_size=6, shuffle=True)
list(dl)

# ## Putting It All Together

weights = init_params((28*28,1))
bias = init_params(1)

dl = DataLoader(dset, batch_size=256)
xb,yb = first(dl)
xb.shape,yb.shape

valid_dl = DataLoader(valid_dset, batch_size=256)

batch = train_x[:4]
batch.shape

preds = linear1(batch)
preds

loss = mnist_loss(preds, train_y[:4])
loss

loss.backward()
weights.grad.shape,weights.grad.mean(),bias.grad


def calc_grad(xb, yb, model):
    preds = model(xb)
    loss = mnist_loss(preds, yb)
    loss.backward()


calc_grad(batch, train_y[:4], linear1)
weights.grad.mean(),bias.grad

calc_grad(batch, train_y[:4], linear1)
weights.grad.mean(),bias.grad

weights.grad.zero_()
bias.grad.zero_();


def train_epoch(model, lr, params):
    for xb,yb in dl:
        calc_grad(xb, yb, model)
        for p in params:
            p.data -= p.grad*lr
            p.grad.zero_()


(preds>0.0).float() == train_y[:4]


def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds>0.5) == yb
    return correct.float().mean()


batch_accuracy(linear1(batch), train_y[:4])


def validate_epoch(model):
    accs = [batch_accuracy(model(xb), yb) for xb,yb in valid_dl]
    return round(torch.stack(accs).mean().item(), 4)


validate_epoch(linear1)

lr = 1.
params = weights,bias
train_epoch(linear1, lr, params)
validate_epoch(linear1)

for i in range(20):
    train_epoch(linear1, lr, params)
    print(validate_epoch(linear1), end=' ')

# ### Creating an Optimizer

linear_model = nn.Linear(28*28,1)

w,b = linear_model.parameters()
w.shape,b.shape


class BasicOptim:
    def __init__(self,params,lr): self.params,self.lr = list(params),lr

    def step(self, *args, **kwargs):
        for p in self.params: p.data -= p.grad.data * self.lr

    def zero_grad(self, *args, **kwargs):
        for p in self.params: p.grad = None


opt = BasicOptim(linear_model.parameters(), lr)


def train_epoch(model):
    for xb,yb in dl:
        calc_grad(xb, yb, model)
        opt.step()
        opt.zero_grad()


validate_epoch(linear_model)


def train_model(model, epochs):
    for i in range(epochs):
        train_epoch(model)
        print(validate_epoch(model), end=' ')


train_model(linear_model, 20)

linear_model = nn.Linear(28*28,1)
opt = SGD(linear_model.parameters(), lr)
train_model(linear_model, 20)

dls = DataLoaders(dl, valid_dl)

learn = Learner(dls, nn.Linear(28*28,1), opt_func=SGD,
                loss_func=mnist_loss, metrics=batch_accuracy)

learn.fit(10, lr=lr)


# ## Adding a Nonlinearity

def simple_net(xb):
    res = xb@w1 + b1
    res = res.max(tensor(0.0))
    res = res@w2 + b2
    return res


w1 = init_params((28*28,30))
b1 = init_params(30)
w2 = init_params((30,1))
b2 = init_params(1)

plot_function(F.relu)

simple_net = nn.Sequential(
    nn.Linear(28*28,30),
    nn.ReLU(),
    nn.Linear(30,1)
)

learn = Learner(dls, simple_net, opt_func=SGD,
                loss_func=mnist_loss, metrics=batch_accuracy)

learn.fit(40, 0.1)

plt.plot(L(learn.recorder.values).itemgot(2));

learn.recorder.values[-1][2]

# ### Going Deeper

dls = ImageDataLoaders.from_folder(path)
learn = cnn_learner(dls, resnet18, pretrained=False,
                    loss_func=F.cross_entropy, metrics=accuracy)
learn.fit_one_cycle(1, 0.1)

# ## Jargon Recap

# ## Questionnaire

# 1. How is a grayscale image represented on a computer? How about a color image?
# 1. How are the files and folders in the `MNIST_SAMPLE` dataset structured? Why?
# 1. Explain how the "pixel similarity" approach to classifying digits works.
# 1. What is a list comprehension? Create one now that selects odd numbers from a list and doubles them.
# 1. What is a "rank-3 tensor"?
# 1. What is the difference between tensor rank and shape? How do you get the rank from the shape?
# 1. What are RMSE and L1 norm?
# 1. How can you apply a calculation on thousands of numbers at once, many thousands of times faster than a Python loop?
# 1. Create a 3×3 tensor or array containing the numbers from 1 to 9. Double it. Select the bottom-right four numbers.
# 1. What is broadcasting?
# 1. Are metrics generally calculated using the training set, or the validation set? Why?
# 1. What is SGD?
# 1. Why does SGD use mini-batches?
# 1. What are the seven steps in SGD for machine learning?
# 1. How do we initialize the weights in a model?
# 1. What is "loss"?
# 1. Why can't we always use a high learning rate?
# 1. What is a "gradient"?
# 1. Do you need to know how to calculate gradients yourself?
# 1. Why can't we use accuracy as a loss function?
# 1. Draw the sigmoid function. What is special about its shape?
# 1. What is the difference between a loss function and a metric?
# 1. What is the function to calculate new weights using a learning rate?
# 1. What does the `DataLoader` class do?
# 1. Write pseudocode showing the basic steps taken in each epoch for SGD.
# 1. Create a function that, if passed two arguments `[1,2,3,4]` and `'abcd'`, returns `[(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]`. What is special about that output data structure?
# 1. What does `view` do in PyTorch?
# 1. What are the "bias" parameters in a neural network? Why do we need them?
# 1. What does the `@` operator do in Python?
# 1. What does the `backward` method do?
# 1. Why do we have to zero the gradients?
# 1. What information do we have to pass to `Learner`?
# 1. Show Python or pseudocode for the basic steps of a training loop.
# 1. What is "ReLU"? Draw a plot of it for values from `-2` to `+2`.
# 1. What is an "activation function"?
# 1. What's the difference between `F.relu` and `nn.ReLU`?
# 1. The universal approximation theorem shows that any function can be approximated as closely as needed using just one nonlinearity. So why do we normally use more?

# ### Further Research

# 1. Create your own implementation of `Learner` from scratch, based on the training loop shown in this chapter.
# 1. Complete all the steps in this chapter using the full MNIST datasets (that is, for all digits, not just 3s and 7s). This is a significant project and will take you quite a bit of time to complete! You'll need to do some of your own research to figure out how to overcome some obstacles you'll meet on the way.
