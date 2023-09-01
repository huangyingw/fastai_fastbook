# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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

# hide
from fastai.vision.all import *
from PIL import Image
import matplotlib.pyplot as plt
from fastbook import *
import fastbook
fastbook.setup_book()


# # Appendix: Jupyter Notebook 101

# ## Introduction

1 + 1

# ## Writing

3 / 2

# ## Modes

# ## Other Important Considerations

# ## Markdown Formatting
#

# ### Italics, Bold, Strikethrough, Inline, Blockquotes and Links

# ### Headings

# ### Lists

# ## Code Capabilities

# Import necessary libraries


a = 1
b = a + 1
c = b + a + 1
d = c + b + a + 1
a, b, c, d

plt.plot([a, b, c, d])
plt.show()

Image.open(image_cat())

# ## Running the App Locally

# ## Creating a Notebook

# ## Shortcuts and Tricks

# ### Command Mode Shortcuts

# ### Cell Tricks

# ### Line Magics

# %timeit [i+1 for i in range(1000)]
