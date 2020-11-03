from fastai.vision.all import *
from fastbook import *
import fastbook
fastbook.setup_book()

path = untar_data(URLs.BIWI_HEAD_POSE)
path.ls().sorted()

(path / '01').ls().sorted()

img_files = get_image_files(path)
def img2pose(x): return Path(f'{str(x)[:-7]}pose.txt')


img2pose(img_files[0])

im = PILImage.create(img_files[0])
im.shape

im.to_thumb(160)

cal = np.genfromtxt(path / '01' / 'rgb.cal', skip_footer=6)
cal


def get_ctr(f):
    ctr = np.genfromtxt(img2pose(f), skip_header=3)
    c1 = ctr[0] * cal[0][0] / ctr[2] + cal[0][2]
    c2 = ctr[1] * cal[1][1] / ctr[2] + cal[1][2]
    return tensor([c1, c2])


get_ctr(img_files[0])

biwi = DataBlock(
    blocks=(ImageBlock, PointBlock),
    get_items=get_image_files,
    get_y=get_ctr,
    splitter=FuncSplitter(lambda o: o.parent.name == '13'),
    batch_tfms=[*aug_transforms(size=(240, 320)),
                Normalize.from_stats(*imagenet_stats)]
)

dls = biwi.dataloaders(path)
dls.show_batch(max_n=9, figsize=(8, 6))

xb, yb = dls.one_batch()
xb.shape, yb.shape

show_image(xb[0])


yb[0]


learn = cnn_learner(dls, resnet18, y_range=(-1, 1))


def sigmoid_range(x, lo, hi): return torch.sigmoid(x) * (hi - lo) + lo


plot_function(partial(sigmoid_range, lo=-1, hi=1), min=-4, max=4)

dls.loss_func

learn.lr_find()

lr = 1e-2
learn.fine_tune(3, lr)

math.sqrt(0.0001)

learn.show_results(ds_idx=1, nrows=3, figsize=(6, 8))
