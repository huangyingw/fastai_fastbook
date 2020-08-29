# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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

# # Appendix: Jupyter Notebook 101

# ## Introduction

1+1

# ## Writing

3/2

# ## Modes

# ## Other Important Considerations

# ## Markdown Formatting
#

# ### Italics, Bold, Strikethrough, Inline, Blockquotes and Links

# ### Headings

# ### Lists

# ## Code Capabilities

# Import necessary libraries
from fastai.vision.all import * 
import matplotlib.pyplot as plt

from PIL import Image

a = 1
b = a + 1
c = b + a + 1
d = c + b + a + 1
a, b, c ,d

plt.plot([a,b,c,d])
plt.show()

Image.open('images/chapter1_cat_example.jpg')

# ## Running the App Locally

# ## Creating a Notebook

# ## Shortcuts and Tricks

# ### Command Mode Shortcuts

# ### Cell Tricks

# ### Line Magics

# %timeit [i+1 for i in range(1000)]
