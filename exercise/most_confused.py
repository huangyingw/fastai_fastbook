from fastai.callback.fp16 import *
from fastai.vision.all import *
from fastbook import *
import fastbook
fastbook.setup_book()

path = untar_data(URLs.PETS)
pets = DataBlock(blocks=(ImageBlock, CategoryBlock),
                 get_items=get_image_files,
                 splitter=RandomSubsetSplitter(train_sz=0.05, valid_sz=0.01, seed=42),
                 get_y=using_attr(RegexLabeller(r'(.+)_\d+.jpg$'), 'name'),
                 item_tfms=Resize(460),
                 batch_tfms=aug_transforms(size=224, min_scale=0.75))
dls = pets.dataloaders(path / "images")

learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(12, 12), dpi=60)

interp.most_confused(min_val=5)

learn.recorder.plot_loss()
