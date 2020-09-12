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
# !pip install -Uqq fastbook
import fastbook
fastbook.setup_book()

#hide
from fastbook import *
from IPython.display import display,HTML

# # Data Munging with fastai's Mid-Level API

# ## Going Deeper into fastai's Layered API

# +
from fastai.text.all import *

dls = TextDataLoaders.from_folder(untar_data(URLs.IMDB), valid='test')
# -

path = untar_data(URLs.IMDB)
dls = DataBlock(
    blocks=(TextBlock.from_folder(path),CategoryBlock),
    get_y = parent_label,
    get_items=partial(get_text_files, folders=['train', 'test']),
    splitter=GrandparentSplitter(valid_name='test')
).dataloaders(path)

# ### Transforms

files = get_text_files(path, folders = ['train', 'test'])
txts = L(o.open().read() for o in files[:2000])

tok = Tokenizer.from_folder(path)
tok.setup(txts)
toks = txts.map(tok)
toks[0]

num = Numericalize()
num.setup(toks)
nums = toks.map(num)
nums[0][:10]

nums_dec = num.decode(nums[0][:10]); nums_dec

tok.decode(nums_dec)

tok((txts[0], txts[1]))


# ### Writing Your Own Transform

def f(x:int): return x+1
tfm = Transform(f)
tfm(2),tfm(2.0)


@Transform
def f(x:int): return x+1
f(2),f(2.0)


class NormalizeMean(Transform):
    def setups(self, items): self.mean = sum(items)/len(items)
    def encodes(self, x): return x-self.mean
    def decodes(self, x): return x+self.mean


tfm = NormalizeMean()
tfm.setup([1,2,3,4,5])
start = 2
y = tfm(start)
z = tfm.decode(y)
tfm.mean,y,z

# ### Pipeline

tfms = Pipeline([tok, num])
t = tfms(txts[0]); t[:20]

tfms.decode(t)[:100]

# ## TfmdLists and Datasets: Transformed Collections

# ### TfmdLists

tls = TfmdLists(files, [Tokenizer.from_folder(path), Numericalize])

t = tls[0]; t[:20]

tls.decode(t)[:100]

tls.show(t)

cut = int(len(files)*0.8)
splits = [list(range(cut)), list(range(cut,len(files)))]
tls = TfmdLists(files, [Tokenizer.from_folder(path), Numericalize], 
                splits=splits)

tls.valid[0][:20]

lbls = files.map(parent_label)
lbls

cat = Categorize()
cat.setup(lbls)
cat.vocab, cat(lbls[0])

tls_y = TfmdLists(files, [parent_label, Categorize()])
tls_y[0]

# ### Datasets

x_tfms = [Tokenizer.from_folder(path), Numericalize]
y_tfms = [parent_label, Categorize()]
dsets = Datasets(files, [x_tfms, y_tfms])
x,y = dsets[0]
x[:20],y

x_tfms = [Tokenizer.from_folder(path), Numericalize]
y_tfms = [parent_label, Categorize()]
dsets = Datasets(files, [x_tfms, y_tfms], splits=splits)
x,y = dsets.valid[0]
x[:20],y

t = dsets.valid[0]
dsets.decode(t)

dls = dsets.dataloaders(bs=64, before_batch=pad_input)

tfms = [[Tokenizer.from_folder(path), Numericalize], [parent_label, Categorize]]
files = get_text_files(path, folders = ['train', 'test'])
splits = GrandparentSplitter(valid_name='test')(files)
dsets = Datasets(files, tfms, splits=splits)
dls = dsets.dataloaders(dl_type=SortedDL, before_batch=pad_input)

path = untar_data(URLs.IMDB)
dls = DataBlock(
    blocks=(TextBlock.from_folder(path),CategoryBlock),
    get_y = parent_label,
    get_items=partial(get_text_files, folders=['train', 'test']),
    splitter=GrandparentSplitter(valid_name='test')
).dataloaders(path)

# ## Applying the Mid-Level Data API: SiamesePair

from fastai.vision.all import *
path = untar_data(URLs.PETS)
files = get_image_files(path/"images")


class SiameseImage(fastuple):
    def show(self, ctx=None, **kwargs): 
        img1,img2,same_breed = self
        if not isinstance(img1, Tensor):
            if img2.size != img1.size: img2 = img2.resize(img1.size)
            t1,t2 = tensor(img1),tensor(img2)
            t1,t2 = t1.permute(2,0,1),t2.permute(2,0,1)
        else: t1,t2 = img1,img2
        line = t1.new_zeros(t1.shape[0], t1.shape[1], 10)
        return show_image(torch.cat([t1,line,t2], dim=2), 
                          title=same_breed, ctx=ctx)


img = PILImage.create(files[0])
s = SiameseImage(img, img, True)
s.show();

img1 = PILImage.create(files[1])
s1 = SiameseImage(img, img1, False)
s1.show();

s2 = Resize(224)(s1)
s2.show();


def label_func(fname):
    return re.match(r'^(.*)_\d+.jpg$', fname.name).groups()[0]


class SiameseTransform(Transform):
    def __init__(self, files, label_func, splits):
        self.labels = files.map(label_func).unique()
        self.lbl2files = {l: L(f for f in files if label_func(f) == l) 
                          for l in self.labels}
        self.label_func = label_func
        self.valid = {f: self._draw(f) for f in files[splits[1]]}
        
    def encodes(self, f):
        f2,t = self.valid.get(f, self._draw(f))
        img1,img2 = PILImage.create(f),PILImage.create(f2)
        return SiameseImage(img1, img2, t)
    
    def _draw(self, f):
        same = random.random() < 0.5
        cls = self.label_func(f)
        if not same: 
            cls = random.choice(L(l for l in self.labels if l != cls)) 
        return random.choice(self.lbl2files[cls]),same


splits = RandomSplitter()(files)
tfm = SiameseTransform(files, label_func, splits)
tfm(files[0]).show();

tls = TfmdLists(files, tfm, splits=splits)
show_at(tls.valid, 0);

dls = tls.dataloaders(after_item=[Resize(224), ToTensor], 
    after_batch=[IntToFloatTensor, Normalize.from_stats(*imagenet_stats)])

# ## Conclusion

# ## Questionnaire

# 1. Why do we say that fastai has a "layered" API? What does it mean?
# 1. Why does a `Transform` have a `decode` method? What does it do?
# 1. Why does a `Transform` have a `setup` method? What does it do?
# 1. How does a `Transform` work when called on a tuple?
# 1. Which methods do you need to implement when writing your own `Transform`?
# 1. Write a `Normalize` transform that fully normalizes items (subtract the mean and divide by the standard deviation of the dataset), and that can decode that behavior. Try not to peek!
# 1. Write a `Transform` that does the numericalization of tokenized texts (it should set its vocab automatically from the dataset seen and have a `decode` method). Look at the source code of fastai if you need help.
# 1. What is a `Pipeline`?
# 1. What is a `TfmdLists`? 
# 1. What is a `Datasets`? How is it different from a `TfmdLists`?
# 1. Why are `TfmdLists` and `Datasets` named with an "s"?
# 1. How can you build a `DataLoaders` from a `TfmdLists` or a `Datasets`?
# 1. How do you pass `item_tfms` and `batch_tfms` when building a `DataLoaders` from a `TfmdLists` or a `Datasets`?
# 1. What do you need to do when you want to have your custom items work with methods like `show_batch` or `show_results`?
# 1. Why can we easily apply fastai data augmentation transforms to the `SiamesePair` we built?

# ### Further Research

# 1. Use the mid-level API to prepare the data in `DataLoaders` on your own datasets. Try this with the Pet dataset and the Adult dataset from Chapter 1.
# 1. Look at the Siamese tutorial in the fastai documentation to learn how to customize the behavior of `show_batch` and `show_results` for new type of items. Implement it in your own project.

# ## Understanding fastai's Applications: Wrap Up

# Congratulations—you've completed all of the chapters in this book that cover the key practical parts of training models and using deep learning! You know how to use all of fastai's built-in applications, and how to customize them using the data block API and loss functions. You even know how to create a neural network from scratch, and train it! (And hopefully you now know some of the questions to ask to make sure your creations help improve society too.)
#
# The knowledge you already have is enough to create full working prototypes of many types of neural network applications. More importantly, it will help you understand the capabilities and limitations of deep learning models, and how to design a system that's well adapted to them.
#
# In the rest of this book we will be pulling apart those applications, piece by piece, to understand the foundations they are built on. This is important knowledge for a deep learning practitioner, because it is what allows you to inspect and debug models that you build and create new applications that are customized for your particular projects.


