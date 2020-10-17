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

# hide
from fastai.vision.all import *
from fastbook import *
import fastbook
fastbook.setup_book()

# hide

# # Application Architectures Deep Dive

# ## Computer Vision

# ### cnn_learner

model_meta[resnet50]

create_head(20, 2)

# ### unet_learner

# ### A Siamese Network

# +
# hide
path = untar_data(URLs.PETS)
files = get_image_files(path / "images")

class SiameseImage(fastuple):
    def show(self, ctx=None, **kwargs):
        img1, img2, same_breed = self
        if not isinstance(img1, Tensor):
            if img2.size != img1.size:
                img2 = img2.resize(img1.size)
            t1, t2 = tensor(img1), tensor(img2)
            t1, t2 = t1.permute(2, 0, 1), t2.permute(2, 0, 1)
        else:
            t1, t2 = img1, img2
        line = t1.new_zeros(t1.shape[0], t1.shape[1], 10)
        return show_image(torch.cat([t1, line, t2], dim=2),
                          title=same_breed, ctx=ctx)

def label_func(fname):
    return re.match(r'^(.*)_\d+.jpg$', fname.name).groups()[0]

class SiameseTransform(Transform):
    def __init__(self, files, label_func, splits):
        self.labels = files.map(label_func).unique()
        self.lbl2files = {l: L(f for f in files if label_func(f) == l) for l in self.labels}
        self.label_func = label_func
        self.valid = {f: self._draw(f) for f in files[splits[1]]}

    def encodes(self, f):
        f2, t = self.valid.get(f, self._draw(f))
        img1, img2 = PILImage.create(f), PILImage.create(f2)
        return SiameseImage(img1, img2, t)

    def _draw(self, f):
        same = random.random() < 0.5
        cls = self.label_func(f)
        if not same:
            cls = random.choice(L(l for l in self.labels if l != cls))
        return random.choice(self.lbl2files[cls]), same

splits = RandomSplitter()(files)
tfm = SiameseTransform(files, label_func, splits)
tls = TfmdLists(files, tfm, splits=splits)
dls = tls.dataloaders(after_item=[Resize(224), ToTensor],
                      after_batch=[IntToFloatTensor, Normalize.from_stats(*imagenet_stats)])


# -

class SiameseModel(Module):
    def __init__(self, encoder, head):
        self.encoder, self.head = encoder, head

    def forward(self, x1, x2):
        ftrs = torch.cat([self.encoder(x1), self.encoder(x2)], dim=1)
        return self.head(ftrs)


encoder = create_body(resnet34, cut=-2)

head = create_head(512 * 4, 2, ps=0.5)

model = SiameseModel(encoder, head)


def loss_func(out, targ):
    return nn.CrossEntropyLoss()(out, targ.long())


def siamese_splitter(model):
    return [params(model.encoder), params(model.head)]


learn = Learner(dls, model, loss_func=loss_func,
                splitter=siamese_splitter, metrics=accuracy)
learn.freeze()

learn.fit_one_cycle(4, 3e-3)

learn.unfreeze()
learn.fit_one_cycle(4, slice(1e-6, 1e-4))

# ## Natural Language Processing

# ## Tabular

# ## Wrapping Up Architectures

# ## Questionnaire

# 1. What is the "head" of a neural net?
# 1. What is the "body" of a neural net?
# 1. What is "cutting" a neural net? Why do we need to do this for transfer learning?
# 1. What is `model_meta`? Try printing it to see what's inside.
# 1. Read the source code for `create_head` and make sure you understand what each line does.
# 1. Look at the output of `create_head` and make sure you understand why each layer is there, and how the `create_head` source created it.
# 1. Figure out how to change the dropout, layer size, and number of layers created by `cnn_learner`, and see if you can find values that result in better accuracy from the pet recognizer.
# 1. What does `AdaptiveConcatPool2d` do?
# 1. What is "nearest neighbor interpolation"? How can it be used to upsample convolutional activations?
# 1. What is a "transposed convolution"? What is another name for it?
# 1. Create a conv layer with `transpose=True` and apply it to an image. Check the output shape.
# 1. Draw the U-Net architecture.
# 1. What is "BPTT for Text Classification" (BPT3C)?
# 1. How do we handle different length sequences in BPT3C?
# 1. Try to run each line of `TabularModel.forward` separately, one line per cell, in a notebook, and look at the input and output shapes at each step.
# 1. How is `self.layers` defined in `TabularModel`?
# 1. What are the five steps for preventing over-fitting?
# 1. Why don't we reduce architecture complexity before trying other approaches to preventing overfitting?

# ### Further Research

# 1. Write your own custom head and try training the pet recognizer with it. See if you can get a better result than fastai's default.
# 1. Try switching between `AdaptiveConcatPool2d` and `AdaptiveAvgPool2d` in a CNN head and see what difference it makes.
# 1. Write your own custom splitter to create a separate parameter group for every ResNet block, and a separate group for the stem. Try training with it, and see if it improves the pet recognizer.
# 1. Read the online chapter about generative image models, and create your own colorizer, super-resolution model, or style transfer model.
# 1. Create a custom head using nearest neighbor interpolation and use it to do segmentation on CamVid.
