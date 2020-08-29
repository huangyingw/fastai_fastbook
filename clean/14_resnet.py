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

# # ResNets

# ## Going Back to Imagenette

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

dls.show_batch(max_n=4)


def avg_pool(x): return x.mean((2,3))


def block(ni, nf): return ConvLayer(ni, nf, stride=2)
def get_model():
    return nn.Sequential(
        block(3, 16),
        block(16, 32),
        block(32, 64),
        block(64, 128),
        block(128, 256),
        nn.AdaptiveAvgPool2d(1),
        Flatten(),
        nn.Linear(256, dls.c))


# +
def get_learner(m):
    return Learner(dls, m, loss_func=nn.CrossEntropyLoss(), metrics=accuracy
                  ).to_fp16()

learn = get_learner(get_model())
# -

learn.lr_find()

learn.fit_one_cycle(5, 3e-3)


# ## Building a Modern CNN: ResNet

# ### Skip Connections

class ResBlock(Module):
    def __init__(self, ni, nf):
        self.convs = nn.Sequential(
            ConvLayer(ni,nf),
            ConvLayer(nf,nf, norm_type=NormType.BatchZero))
        
    def forward(self, x): return x + self.convs(x)


def _conv_block(ni,nf,stride):
    return nn.Sequential(
        ConvLayer(ni, nf, stride=stride),
        ConvLayer(nf, nf, act_cls=None, norm_type=NormType.BatchZero))


class ResBlock(Module):
    def __init__(self, ni, nf, stride=1):
        self.convs = _conv_block(ni,nf,stride)
        self.idconv = noop if ni==nf else ConvLayer(ni, nf, 1, act_cls=None)
        self.pool = noop if stride==1 else nn.AvgPool2d(2, ceil_mode=True)

    def forward(self, x):
        return F.relu(self.convs(x) + self.idconv(self.pool(x)))


def block(ni,nf): return ResBlock(ni, nf, stride=2)
learn = get_learner(get_model())

learn.fit_one_cycle(5, 3e-3)


def block(ni, nf):
    return nn.Sequential(ResBlock(ni, nf, stride=2), ResBlock(nf, nf))


learn = get_learner(get_model())
learn.fit_one_cycle(5, 3e-3)


# ### A State-of-the-Art ResNet

def _resnet_stem(*sizes):
    return [
        ConvLayer(sizes[i], sizes[i+1], 3, stride = 2 if i==0 else 1)
            for i in range(len(sizes)-1)
    ] + [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]


_resnet_stem(3,32,32,64)


class ResNet(nn.Sequential):
    def __init__(self, n_out, layers, expansion=1):
        stem = _resnet_stem(3,32,32,64)
        self.block_szs = [64, 64, 128, 256, 512]
        for i in range(1,5): self.block_szs[i] *= expansion
        blocks = [self._make_layer(*o) for o in enumerate(layers)]
        super().__init__(*stem, *blocks,
                         nn.AdaptiveAvgPool2d(1), Flatten(),
                         nn.Linear(self.block_szs[-1], n_out))
    
    def _make_layer(self, idx, n_layers):
        stride = 1 if idx==0 else 2
        ch_in,ch_out = self.block_szs[idx:idx+2]
        return nn.Sequential(*[
            ResBlock(ch_in if i==0 else ch_out, ch_out, stride if i==0 else 1)
            for i in range(n_layers)
        ])


rn = ResNet(dls.c, [2,2,2,2])

learn = get_learner(rn)
learn.fit_one_cycle(5, 3e-3)


# ### Bottleneck Layers

def _conv_block(ni,nf,stride):
    return nn.Sequential(
        ConvLayer(ni, nf//4, 1),
        ConvLayer(nf//4, nf//4, stride=stride), 
        ConvLayer(nf//4, nf, 1, act_cls=None, norm_type=NormType.BatchZero))


dls = get_data(URLs.IMAGENETTE_320, presize=320, resize=224)

rn = ResNet(dls.c, [3,4,6,3], 4)

learn = get_learner(rn)
learn.fit_one_cycle(20, 3e-3)

# ## Conclusion

# ## Questionnaire

# 1. How did we get to a single vector of activations in the CNNs used for MNIST in previous chapters? Why isn't that suitable for Imagenette?
# 1. What do we do for Imagenette instead?
# 1. What is "adaptive pooling"?
# 1. What is "average pooling"?
# 1. Why do we need `Flatten` after an adaptive average pooling layer?
# 1. What is a "skip connection"?
# 1. Why do skip connections allow us to train deeper models?
# 1. What does <<resnet_depth>> show? How did that lead to the idea of skip connections?
# 1. What is "identity mapping"?
# 1. What is the basic equation for a ResNet block (ignoring batchnorm and ReLU layers)?
# 1. What do ResNets have to do with residuals?
# 1. How do we deal with the skip connection when there is a stride-2 convolution? How about when the number of filters changes?
# 1. How can we express a 1×1 convolution in terms of a vector dot product?
# 1. Create a `1x1 convolution` with `F.conv2d` or `nn.Conv2d` and apply it to an image. What happens to the `shape` of the image?
# 1. What does the `noop` function return?
# 1. Explain what is shown in <<resnet_surface>>.
# 1. When is top-5 accuracy a better metric than top-1 accuracy?
# 1. What is the "stem" of a CNN?
# 1. Why do we use plain convolutions in the CNN stem, instead of ResNet blocks?
# 1. How does a bottleneck block differ from a plain ResNet block?
# 1. Why is a bottleneck block faster?
# 1. How do fully convolutional nets (and nets with adaptive pooling in general) allow for progressive resizing?

# ### Further Research

# 1. Try creating a fully convolutional net with adaptive average pooling for MNIST (note that you'll need fewer stride-2 layers). How does it compare to a network without such a pooling layer?
# 1. In <<chapter_foundations>> we introduce *Einstein summation notation*. Skip ahead to see how this works, and then write an implementation of the 1×1 convolution operation using `torch.einsum`. Compare it to the same operation using `torch.conv2d`.
# 1. Write a "top-5 accuracy" function using plain PyTorch or plain Python.
# 1. Train a model on Imagenette for more epochs, with and without label smoothing. Take a look at the Imagenette leaderboards and see how close you can get to the best results shown. Read the linked pages describing the leading approaches.


