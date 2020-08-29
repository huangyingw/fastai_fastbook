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
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + hide_input=false
#hide
from utils import *


# -

# # The Training Process

# ## Establishing a Baseline

def get_data(url, presize, resize):
    path = untar_data(url)
    return DataBlock(
        blocks=(ImageBlock, CategoryBlock), get_items=get_image_files, 
        splitter=GrandparentSplitter(valid_name='val'),
        get_y=parent_label, item_tfms=Resize(presize),
        batch_tfms=[*aug_transforms(min_scale=0.5, size=resize),
                    Normalize.from_stats(*imagenet_stats)],
    ).dataloaders(path, bs=128)


dls = get_data(URLs.IMAGENETTE_160, 160, 128)


def get_learner(**kwargs):
    return cnn_learner(dls, resnet34, pretrained=False,
                    metrics=accuracy, **kwargs).to_fp16()


learn = get_learner()
learn.fit_one_cycle(3, 0.003)

learn = get_learner(opt_func=SGD)

learn.lr_find()

learn.fit_one_cycle(3, 0.03, moms=(0,0,0))


# ## A Generic Optimizer

def sgd_cb(p, lr, **kwargs): p.data.add_(-lr, p.grad.data)


opt_func = partial(Optimizer, cbs=[sgd_cb])

learn = get_learner(opt_func=opt_func)
learn.fit(3, 0.03)

# ## Momentum

# + hide_input=true
x = np.linspace(-4, 4, 100)
y = 1 - (x/3) ** 2
x1 = x + np.random.randn(100) * 0.1
y1 = y + np.random.randn(100) * 0.1
plt.scatter(x1,y1)
idx = x1.argsort()
beta,avg,res = 0.7,0,[]
for i in idx:
    avg = beta * avg + (1-beta) * y1[i]
    res.append(avg/(1-beta**(i+1)))
plt.plot(x1[idx],np.array(res), color='red');

# + hide_input=true
x = np.linspace(-4, 4, 100)
y = 1 - (x/3) ** 2
x1 = x + np.random.randn(100) * 0.1
y1 = y + np.random.randn(100) * 0.1
_,axs = plt.subplots(2,2, figsize=(12,8))
betas = [0.5,0.7,0.9,0.99]
idx = x1.argsort()
for beta,ax in zip(betas, axs.flatten()):
    ax.scatter(x1,y1)
    avg,res = 0,[]
    for i in idx:
        avg = beta * avg + (1-beta) * y1[i]
        res.append(avg)#/(1-beta**(i+1)))
    ax.plot(x1[idx],np.array(res), color='red');
    ax.set_title(f'beta={beta}')


# -

def average_grad(p, mom, grad_avg=None, **kwargs):
    if grad_avg is None: grad_avg = torch.zeros_like(p.grad.data)
    return {'grad_avg': grad_avg*mom + p.grad.data}


def momentum_step(p, lr, grad_avg, **kwargs): p.data.add_(-lr, grad_avg)


opt_func = partial(Optimizer, cbs=[average_grad,momentum_step], mom=0.9)

learn = get_learner(opt_func=opt_func)
learn.fit_one_cycle(3, 0.03)

learn.recorder.plot_sched()


# ## RMSProp

def average_sqr_grad(p, sqr_mom, sqr_avg=None, **kwargs):
    if sqr_avg is None: sqr_avg = torch.zeros_like(p.grad.data)
    return {'sqr_avg': sqr_avg*sqr_mom + p.grad.data**2}


# +
def rms_prop_step(p, lr, sqr_avg, eps, grad_avg=None, **kwargs):
    denom = sqr_avg.sqrt().add_(eps)
    p.data.addcdiv_(-lr, p.grad, denom)

opt_func = partial(Optimizer, cbs=[average_sqr_grad,rms_prop_step],
                   sqr_mom=0.99, eps=1e-7)
# -

learn = get_learner(opt_func=opt_func)
learn.fit_one_cycle(3, 0.003)


# ## Adam

# ## Decoupled Weight Decay

# ## Callbacks

# ### Creating a Callback

class ModelResetter(Callback):
    def begin_train(self):    self.model.reset()
    def begin_validate(self): self.model.reset()


class RNNRegularizer(Callback):
    def __init__(self, alpha=0., beta=0.): self.alpha,self.beta = alpha,beta

    def after_pred(self):
        self.raw_out,self.out = self.pred[1],self.pred[2]
        self.learn.pred = self.pred[0]

    def after_loss(self):
        if not self.training: return
        if self.alpha != 0.:
            self.learn.loss += self.alpha * self.out[-1].float().pow(2).mean()
        if self.beta != 0.:
            h = self.raw_out[-1]
            if len(h)>1:
                self.learn.loss += self.beta * (h[:,1:] - h[:,:-1]
                                               ).float().pow(2).mean()


# ### Callback Ordering and Exceptions

class TerminateOnNaNCallback(Callback):
    run_before=Recorder
    def after_batch(self):
        if torch.isinf(self.loss) or torch.isnan(self.loss):
            raise CancelFitException

# ## Conclusion

# ## Questionnaire

# 1. What is the equation for a step of SGD, in math or code (as you prefer)?
# 1. What do we pass to `cnn_learner` to use a non-default optimizer?
# 1. What are optimizer callbacks?
# 1. What does `zero_grad` do in an optimizer?
# 1. What does `step` do in an optimizer? How is it implemented in the general optimizer?
# 1. Rewrite `sgd_cb` to use the `+=` operator, instead of `add_`.
# 1. What is "momentum"? Write out the equation.
# 1. What's a physical analogy for momentum? How does it apply in our model training settings?
# 1. What does a bigger value for momentum do to the gradients?
# 1. What are the default values of momentum for 1cycle training?
# 1. What is RMSProp? Write out the equation.
# 1. What do the squared values of the gradients indicate?
# 1. How does Adam differ from momentum and RMSProp?
# 1. Write out the equation for Adam.
# 1. Calculate the values of `unbias_avg` and `w.avg` for a few batches of dummy values.
# 1. What's the impact of having a high `eps` in Adam?
# 1. Read through the optimizer notebook in fastai's repo, and execute it.
# 1. In what situations do dynamic learning rate methods like Adam change the behavior of weight decay?
# 1. What are the four steps of a training loop?
# 1. Why is using callbacks better than writing a new training loop for each tweak you want to add?
# 1. What aspects of the design of fastai's callback system make it as flexible as copying and pasting bits of code?
# 1. How can you get the list of events available to you when writing a callback?
# 1. Write the `ModelResetter` callback (without peeking).
# 1. How can you access the necessary attributes of the training loop inside a callback? When can you use or not use the shortcuts that go with them?
# 1. How can a callback influence the control flow of the training loop.
# 1. Write the `TerminateOnNaN` callback (without peeking, if possible).
# 1. How do you make sure your callback runs after or before another callback?

# ### Further Research

# 1. Look up the "Rectified Adam" paper, implement it using the general optimizer framework, and try it out. Search for other recent optimizers that work well in practice, and pick one to implement.
# 1. Look at the mixed-precision callback with the documentation. Try to understand what each event and line of code does.
# 1. Implement your own version of ther learning rate finder from scratch. Compare it with fastai's version.
# 1. Look at the source code of the callbacks that ship with fastai. See if you can find one that's similar to what you're looking to do, to get some inspiration.

# ## Foundations of Deep Learning: Wrap up

# Congratulations, you have made it to the end of the "foundations of deep learning" section of the book! You now understand how all of fastai's applications and most important architectures are built, and the recommended ways to train them—and you have all the information you need to build these from scratch. While you probably won't need to create your own training loop, or batchnorm layer, for instance, knowing what is going on behind the scenes is very helpful for debugging, profiling, and deploying your solutions.
#
# Since you understand the foundations of fastai's applications now, be sure to spend some time digging through the source notebooks and running and experimenting with parts of them. This will give you a better idea of how everything in fastai is developed.
#
# In the next section, we will be looking even further under the covers: we'll explore how the actual forward and backward passes of a neural network are done, and we will see what tools are at our disposal to get better performance. We will then continue with a project that brings together all the material in the book, which we will use to build a tool for interpreting convolutional neural networks. Last but not least, we'll finish by building fastai's `Learner` class from scratch.


