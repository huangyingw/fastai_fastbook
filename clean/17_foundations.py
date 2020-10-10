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

#hide
from fastai.gen_doc.nbdoc import *

# # A Neural Net from the Foundations

# ## Building a Neural Net Layer from Scratch

# ### Modeling a Neuron

# ### Matrix Multiplication from Scratch

import torch
from torch import tensor


def matmul(a,b):
    ar,ac = a.shape # n_rows * n_cols
    br,bc = b.shape
    assert ac==br
    c = torch.zeros(ar, bc)
    for i in range(ar):
        for j in range(bc):
            for k in range(ac): c[i,j] += a[i,k] * b[k,j]
    return c


m1 = torch.randn(5,28*28)
m2 = torch.randn(784,10)

# %time t1=matmul(m1, m2)

# %timeit -n 20 t2=m1@m2

# ### Elementwise Arithmetic

a = tensor([10., 6, -4])
b = tensor([2., 8, 7])
a + b

a < b

(a < b).all(), (a==b).all()

(a + b).mean().item()

m = tensor([[1., 2, 3], [4,5,6], [7,8,9]])
m*m

n = tensor([[1., 2, 3], [4,5,6]])
m*n


def matmul(a,b):
    ar,ac = a.shape
    br,bc = b.shape
    assert ac==br
    c = torch.zeros(ar, bc)
    for i in range(ar):
        for j in range(bc): c[i,j] = (a[i] * b[:,j]).sum()
    return c


# %timeit -n 20 t3 = matmul(m1,m2)

# ### Broadcasting

# #### Broadcasting with a scalar

a = tensor([10., 6, -4])
a > 0

m = tensor([[1., 2, 3], [4,5,6], [7,8,9]])
(m - 5) / 2.73

# #### Broadcasting a vector to a matrix

c = tensor([10.,20,30])
m = tensor([[1., 2, 3], [4,5,6], [7,8,9]])
m.shape,c.shape

m + c

c.expand_as(m)

t = c.expand_as(m)
t.storage()

t.stride(), t.shape

c + m

c = tensor([10.,20,30])
m = tensor([[1., 2, 3], [4,5,6]])
c+m

c = tensor([10.,20])
m = tensor([[1., 2, 3], [4,5,6]])
c+m

c = tensor([10.,20,30])
m = tensor([[1., 2, 3], [4,5,6], [7,8,9]])
c = c.unsqueeze(1)
m.shape,c.shape

c+m

t = c.expand_as(m)
t.storage()

t.stride(), t.shape

c = tensor([10.,20,30])
c.shape, c.unsqueeze(0).shape,c.unsqueeze(1).shape

c.shape, c[None,:].shape,c[:,None].shape

c[None].shape,c[...,None].shape


def matmul(a,b):
    ar,ac = a.shape
    br,bc = b.shape
    assert ac==br
    c = torch.zeros(ar, bc)
    for i in range(ar):
#       c[i,j] = (a[i,:]          * b[:,j]).sum() # previous
        c[i]   = (a[i  ].unsqueeze(-1) * b).sum(dim=0)
    return c


# %timeit -n 20 t4 = matmul(m1,m2)

# #### Broadcasting rules

# ### Einstein Summation

def matmul(a,b): return torch.einsum('ik,kj->ij', a, b)


# %timeit -n 20 t5 = matmul(m1,m2)

# ## The Forward and Backward Passes

# ### Defining and Initializing a Layer

def lin(x, w, b): return x @ w + b


x = torch.randn(200, 100)
y = torch.randn(200)

w1 = torch.randn(100,50)
b1 = torch.zeros(50)
w2 = torch.randn(50,1)
b2 = torch.zeros(1)

l1 = lin(x, w1, b1)
l1.shape

l1.mean(), l1.std()

x = torch.randn(200, 100)
for i in range(50): x = x @ torch.randn(100,100)
x[0:5,0:5]

x = torch.randn(200, 100)
for i in range(50): x = x @ (torch.randn(100,100) * 0.01)
x[0:5,0:5]

x = torch.randn(200, 100)
for i in range(50): x = x @ (torch.randn(100,100) * 0.1)
x[0:5,0:5]

x.std()

x = torch.randn(200, 100)
y = torch.randn(200)

from math import sqrt
w1 = torch.randn(100,50) / sqrt(100)
b1 = torch.zeros(50)
w2 = torch.randn(50,1) / sqrt(50)
b2 = torch.zeros(1)

l1 = lin(x, w1, b1)
l1.mean(),l1.std()


def relu(x): return x.clamp_min(0.)


l2 = relu(l1)
l2.mean(),l2.std()

x = torch.randn(200, 100)
for i in range(50): x = relu(x @ (torch.randn(100,100) * 0.1))
x[0:5,0:5]

x = torch.randn(200, 100)
for i in range(50): x = relu(x @ (torch.randn(100,100) * sqrt(2/100)))
x[0:5,0:5]

x = torch.randn(200, 100)
y = torch.randn(200)

w1 = torch.randn(100,50) * sqrt(2 / 100)
b1 = torch.zeros(50)
w2 = torch.randn(50,1) * sqrt(2 / 50)
b2 = torch.zeros(1)

l1 = lin(x, w1, b1)
l2 = relu(l1)
l2.mean(), l2.std()


def model(x):
    l1 = lin(x, w1, b1)
    l2 = relu(l1)
    l3 = lin(l2, w2, b2)
    return l3


out = model(x)
out.shape


def mse(output, targ): return (output.squeeze(-1) - targ).pow(2).mean()


loss = mse(out, y)


# ### Gradients and the Backward Pass

def mse_grad(inp, targ):
    # grad of loss with respect to output of previous layer
    inp.g = 2. * (inp.squeeze() - targ).unsqueeze(-1) / inp.shape[0]


def relu_grad(inp, out):
    # grad of relu with respect to input activations
    inp.g = (inp>0).float() * out.g


def lin_grad(inp, out, w, b):
    # grad of matmul with respect to input
    inp.g = out.g @ w.t()
    w.g = inp.t() @ out.g
    b.g = out.g.sum(0)


# ### Sidebar: SymPy

from sympy import symbols,diff
sx,sy = symbols('sx sy')
diff(sx**2, sx)


# ### End sidebar

def forward_and_backward(inp, targ):
    # forward pass:
    l1 = inp @ w1 + b1
    l2 = relu(l1)
    out = l2 @ w2 + b2
    # we don't actually need the loss in backward!
    loss = mse(out, targ)

    # backward pass:
    mse_grad(out, targ)
    lin_grad(l2, out, w2, b2)
    relu_grad(l1, l2)
    lin_grad(inp, l1, w1, b1)


# ### Refactoring the Model

class Relu():
    def __call__(self, inp):
        self.inp = inp
        self.out = inp.clamp_min(0.)
        return self.out

    def backward(self): self.inp.g = (self.inp>0).float() * self.out.g


class Lin():
    def __init__(self, w, b): self.w,self.b = w,b

    def __call__(self, inp):
        self.inp = inp
        self.out = inp@self.w + self.b
        return self.out

    def backward(self):
        self.inp.g = self.out.g @ self.w.t()
        self.w.g = self.inp.t() @ self.out.g
        self.b.g = self.out.g.sum(0)


class Mse():
    def __call__(self, inp, targ):
        self.inp = inp
        self.targ = targ
        self.out = (inp.squeeze() - targ).pow(2).mean()
        return self.out

    def backward(self):
        x = (self.inp.squeeze()-self.targ).unsqueeze(-1)
        self.inp.g = 2.*x/self.targ.shape[0]


class Model():
    def __init__(self, w1, b1, w2, b2):
        self.layers = [Lin(w1,b1), Relu(), Lin(w2,b2)]
        self.loss = Mse()

    def __call__(self, x, targ):
        for l in self.layers: x = l(x)
        return self.loss(x, targ)

    def backward(self):
        self.loss.backward()
        for l in reversed(self.layers): l.backward()


model = Model(w1, b1, w2, b2)

loss = model(x, y)

model.backward()


# ### Going to PyTorch

class LayerFunction():
    def __call__(self, *args):
        self.args = args
        self.out = self.forward(*args)
        return self.out

    def forward(self):  raise Exception('not implemented')
    def bwd(self):      raise Exception('not implemented')
    def backward(self): self.bwd(self.out, *self.args)


class Relu(LayerFunction):
    def forward(self, inp): return inp.clamp_min(0.)
    def bwd(self, out, inp): inp.g = (inp>0).float() * out.g


class Lin(LayerFunction):
    def __init__(self, w, b): self.w,self.b = w,b

    def forward(self, inp): return inp@self.w + self.b

    def bwd(self, out, inp):
        inp.g = out.g @ self.w.t()
        self.w.g = self.inp.t() @ self.out.g
        self.b.g = out.g.sum(0)


class Mse(LayerFunction):
    def forward (self, inp, targ): return (inp.squeeze() - targ).pow(2).mean()
    def bwd(self, out, inp, targ):
        inp.g = 2*(inp.squeeze()-targ).unsqueeze(-1) / targ.shape[0]


# +
from torch.autograd import Function

class MyRelu(Function):
    @staticmethod
    def forward(ctx, i):
        result = i.clamp_min(0.)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i, = ctx.saved_tensors
        return grad_output * (i>0).float()


# +
import torch.nn as nn

class LinearLayer(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_out, n_in) * sqrt(2/n_in))
        self.bias = nn.Parameter(torch.zeros(n_out))

    def forward(self, x): return x @ self.weight.t() + self.bias


# -

lin = LinearLayer(10,2)
p1,p2 = lin.parameters()
p1.shape,p2.shape


class Model(nn.Module):
    def __init__(self, n_in, nh, n_out):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_in,nh), nn.ReLU(), nn.Linear(nh,n_out))
        self.loss = mse

    def forward(self, x, targ): return self.loss(self.layers(x).squeeze(), targ)


class Model(Module):
    def __init__(self, n_in, nh, n_out):
        self.layers = nn.Sequential(
            nn.Linear(n_in,nh), nn.ReLU(), nn.Linear(nh,n_out))
        self.loss = mse

    def forward(self, x, targ): return self.loss(self.layers(x).squeeze(), targ)

# ## Conclusion

# ## Questionnaire

# 1. Write the Python code to implement a single neuron.
# 1. Write the Python code to implement ReLU.
# 1. Write the Python code for a dense layer in terms of matrix multiplication.
# 1. Write the Python code for a dense layer in plain Python (that is, with list comprehensions and functionality built into Python).
# 1. What is the "hidden size" of a layer?
# 1. What does the `t` method do in PyTorch?
# 1. Why is matrix multiplication written in plain Python very slow?
# 1. In `matmul`, why is `ac==br`?
# 1. In Jupyter Notebook, how do you measure the time taken for a single cell to execute?
# 1. What is "elementwise arithmetic"?
# 1. Write the PyTorch code to test whether every element of `a` is greater than the corresponding element of `b`.
# 1. What is a rank-0 tensor? How do you convert it to a plain Python data type?
# 1. What does this return, and why? `tensor([1,2]) + tensor([1])`
# 1. What does this return, and why? `tensor([1,2]) + tensor([1,2,3])`
# 1. How does elementwise arithmetic help us speed up `matmul`?
# 1. What are the broadcasting rules?
# 1. What is `expand_as`? Show an example of how it can be used to match the results of broadcasting.
# 1. How does `unsqueeze` help us to solve certain broadcasting problems?
# 1. How can we use indexing to do the same operation as `unsqueeze`?
# 1. How do we show the actual contents of the memory used for a tensor?
# 1. When adding a vector of size 3 to a matrix of size 3×3, are the elements of the vector added to each row or each column of the matrix? (Be sure to check your answer by running this code in a notebook.)
# 1. Do broadcasting and `expand_as` result in increased memory use? Why or why not?
# 1. Implement `matmul` using Einstein summation.
# 1. What does a repeated index letter represent on the left-hand side of einsum?
# 1. What are the three rules of Einstein summation notation? Why?
# 1. What are the forward pass and backward pass of a neural network?
# 1. Why do we need to store some of the activations calculated for intermediate layers in the forward pass?
# 1. What is the downside of having activations with a standard deviation too far away from 1?
# 1. How can weight initialization help avoid this problem?
# 1. What is the formula to initialize weights such that we get a standard deviation of 1 for a plain linear layer, and for a linear layer followed by ReLU?
# 1. Why do we sometimes have to use the `squeeze` method in loss functions?
# 1. What does the argument to the `squeeze` method do? Why might it be important to include this argument, even though PyTorch does not require it?
# 1. What is the "chain rule"? Show the equation in either of the two forms presented in this chapter.
# 1. Show how to calculate the gradients of `mse(lin(l2, w2, b2), y)` using the chain rule.
# 1. What is the gradient of ReLU? Show it in math or code. (You shouldn't need to commit this to memory—try to figure it using your knowledge of the shape of the function.)
# 1. In what order do we need to call the `*_grad` functions in the backward pass? Why?
# 1. What is `__call__`?
# 1. What methods must we implement when writing a `torch.autograd.Function`?
# 1. Write `nn.Linear` from scratch, and test it works.
# 1. What is the difference between `nn.Module` and fastai's `Module`?

# ### Further Research

# 1. Implement ReLU as a `torch.autograd.Function` and train a model with it.
# 1. If you are mathematically inclined, find out what the gradients of a linear layer are in mathematical notation. Map that to the implementation we saw in this chapter.
# 1. Learn about the `unfold` method in PyTorch, and use it along with matrix multiplication to implement your own 2D convolution function. Then train a CNN that uses it.
# 1. Implement everything in this chapter using NumPy instead of PyTorch.
