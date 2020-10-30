from fastai.callback.fp16 import *
from fastai.vision.all import *
from fastbook import *
import fastbook
fastbook.setup_book()


path = untar_data(URLs.PETS)

Path.BASE_PATH = path

path.ls()

(path / "images").ls()

fname = (path / "images").ls()[0]

re.findall(r'(.+)_\d+.jpg$', fname.name)

pets = DataBlock(blocks=(ImageBlock, CategoryBlock),
                 get_items=get_image_files,
                 splitter=RandomSplitter(seed=42),
                 get_y=using_attr(RegexLabeller(r'(.+)_\d+.jpg$'), 'name'),
                 item_tfms=Resize(460),
                 batch_tfms=aug_transforms(size=224, min_scale=0.75))
dls = pets.dataloaders(path / "images")


# +
dblock1 = DataBlock(blocks=(ImageBlock(), CategoryBlock()),
                    get_y=parent_label,
                    item_tfms=Resize(460))
dls1 = dblock1.dataloaders([(Path.cwd() / 'images' / 'grizzly.jpg')] * 100, bs=8)
dls1.train.get_idxs = lambda: Inf.ones
x, y = dls1.valid.one_batch()
_, axs = subplots(1, 2)

x1 = TensorImage(x.clone())
x1 = x1.affine_coord(sz=224)
x1 = x1.rotate(draw=30, p=1.)
x1 = x1.zoom(draw=1.2, p=1.)
x1 = x1.warp(draw_x=-0.2, draw_y=0.2, p=1.)

tfms = setup_aug_tfms([Rotate(draw=30, p=1, size=224), Zoom(draw=1.2, p=1., size=224),
                       Warp(draw_x=-0.2, draw_y=0.2, p=1., size=224)])
x = Pipeline(tfms)(x)
TensorImage(x[0]).show(ctx=axs[0])
TensorImage(x1[0]).show(ctx=axs[1])
