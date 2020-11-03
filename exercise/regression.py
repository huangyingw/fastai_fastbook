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
