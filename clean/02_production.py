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

# hide
from fastai.vision.widgets import *
from fastbook import *
import fastbook
fastbook.setup_book()

# hide

# # From Model to Production

# ## The Practice of Deep Learning

# ### Starting Your Project

# ### The State of Deep Learning

# #### Computer vision

# #### Text (natural language processing)

# #### Combining text and images

# #### Tabular data

# #### Recommendation systems

# #### Other data types

# ### The Drivetrain Approach

# ## Gathering Data

# # clean
# To download images with Bing Image Search, sign up at Microsoft for a free account. You will be given a key, which you can copy and enter in a cell as follows (replacing 'XXX' with your key and executing it):

key = os.environ.get('AZURE_SEARCH_KEY', 'XXX')

search_images_bing

results = search_images_bing(key, 'grizzly bear')
ims = results.attrgot('content_url')
len(ims)

# hide
ims = ['http://3.bp.blogspot.com/-S1scRCkI3vY/UHzV2kucsPI/AAAAAAAAA-k/YQ5UzHEm9Ss/s1600/Grizzly%2BBear%2BWildlife.jpg']

dest = 'images/grizzly.jpg'
download_url(ims[0], dest)

im = Image.open(dest)
im.to_thumb(128, 128)

bear_types = 'grizzly', 'black', 'teddy'
path = Path('bears')

if not path.exists():
    path.mkdir()
    for o in bear_types:
        dest = (path / o)
        dest.mkdir(exist_ok=True)
        results = search_images_bing(key, f'{o} bear')
        download_images(dest, urls=results.attrgot('content_url'))

fns = get_image_files(path)
fns

failed = verify_images(fns)
failed

failed.map(Path.unlink)

# ### Sidebar: Getting Help in Jupyter Notebooks

# ### End sidebar

# ## From Data to DataLoaders

bears = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))

dls = bears.dataloaders(path)

dls.valid.show_batch(max_n=4, nrows=1)

bears = bears.new(item_tfms=Resize(128, ResizeMethod.Squish))
dls = bears.dataloaders(path)
dls.valid.show_batch(max_n=4, nrows=1)

bears = bears.new(item_tfms=Resize(128, ResizeMethod.Pad, pad_mode='zeros'))
dls = bears.dataloaders(path)
dls.valid.show_batch(max_n=4, nrows=1)

bears = bears.new(item_tfms=RandomResizedCrop(128, min_scale=0.3))
dls = bears.dataloaders(path)
dls.train.show_batch(max_n=4, nrows=1, unique=True)

# ### Data Augmentation

bears = bears.new(item_tfms=Resize(128), batch_tfms=aug_transforms(mult=2))
dls = bears.dataloaders(path)
dls.train.show_batch(max_n=8, nrows=2, unique=True)

# ## Training Your Model, and Using It to Clean Your Data

bears = bears.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms())
dls = bears.dataloaders(path)

learn = cnn_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4)

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

interp.plot_top_losses(5, nrows=1)

cleaner = ImageClassifierCleaner(learn)
cleaner

# +
# hide
# for idx in cleaner.delete(): cleaner.fns[idx].unlink()
# for idx,cat in cleaner.change(): shutil.move(str(cleaner.fns[idx]), path/cat)
# -

# ## Turning Your Model into an Online Application

# ### Using the Model for Inference

learn.export()

path = Path()
path.ls(file_exts='.pkl')

learn_inf = load_learner(path / 'export.pkl')

learn_inf.predict('images/grizzly.jpg')

learn_inf.dls.vocab

# ### Creating a Notebook App from the Model

btn_upload = widgets.FileUpload()
btn_upload

# hide
# For the book, we can't actually click an upload button, so we fake it
btn_upload = SimpleNamespace(data=['images/grizzly.jpg'])

img = PILImage.create(btn_upload.data[-1])

out_pl = widgets.Output()
out_pl.clear_output()
with out_pl:
    display(img.to_thumb(128, 128))
out_pl

pred, pred_idx, probs = learn_inf.predict(img)

lbl_pred = widgets.Label()
lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'
lbl_pred

btn_run = widgets.Button(description='Classify')
btn_run


# +
def on_click_classify(change):
    img = PILImage.create(btn_upload.data[-1])
    out_pl.clear_output()
    with out_pl:
        display(img.to_thumb(128, 128))
    pred, pred_idx, probs = learn_inf.predict(img)
    lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'


btn_run.on_click(on_click_classify)
# -

# hide
# Putting back btn_upload to a widget for next cell
btn_upload = widgets.FileUpload()

VBox([widgets.Label('Select your bear!'),
      btn_upload, btn_run, out_pl, lbl_pred])

# ### Turning Your Notebook into a Real App

# +
# hide
# # !jupyter serverextension enable voila —sys-prefix
# -

# ### Deploying your app

# ## How to Avoid Disaster

# ### Unforeseen Consequences and Feedback Loops

# ## Get Writing!

# ## Questionnaire

# 1. Provide an example of where the bear classification model might work poorly in production, due to structural or style differences in the training data.
# 1. Where do text models currently have a major deficiency?
# 1. What are possible negative societal implications of text generation models?
# 1. In situations where a model might make mistakes, and those mistakes could be harmful, what is a good alternative to automating a process?
# 1. What kind of tabular data is deep learning particularly good at?
# 1. What's a key downside of directly using a deep learning model for recommendation systems?
# 1. What are the steps of the Drivetrain Approach?
# 1. How do the steps of the Drivetrain Approach map to a recommendation system?
# 1. Create an image recognition model using data you curate, and deploy it on the web.
# 1. What is `DataLoaders`?
# 1. What four things do we need to tell fastai to create `DataLoaders`?
# 1. What does the `splitter` parameter to `DataBlock` do?
# 1. How do we ensure a random split always gives the same validation set?
# 1. What letters are often used to signify the independent and dependent variables?
# 1. What's the difference between the crop, pad, and squish resize approaches? When might you choose one over the others?
# 1. What is data augmentation? Why is it needed?
# 1. What is the difference between `item_tfms` and `batch_tfms`?
# 1. What is a confusion matrix?
# 1. What does `export` save?
# 1. What is it called when we use a model for getting predictions, instead of training?
# 1. What are IPython widgets?
# 1. When might you want to use CPU for deployment? When might GPU be better?
# 1. What are the downsides of deploying your app to a server, instead of to a client (or edge) device such as a phone or PC?
# 1. What are three examples of problems that could occur when rolling out a bear warning system in practice?
# 1. What is "out-of-domain data"?
# 1. What is "domain shift"?
# 1. What are the three steps in the deployment process?

# ### Further Research

# 1. Consider how the Drivetrain Approach maps to a project or problem you're interested in.
# 1. When might it be best to avoid certain types of data augmentation?
# 1. For a project you're interested in applying deep learning to, consider the thought experiment "What would happen if it went really, really well?"
# 1. Start a blog, and write your first blog post. For instance, write about what you think deep learning might be useful for in a domain you're interested in.
