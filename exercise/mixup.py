from fastai.vision.all import *
from fastbook import *
import fastbook
fastbook.setup_book()

path = untar_data(URLs.IMAGENETTE)

church = PILImage.create(get_image_files_sorted(path / 'train' / 'n03028079')[0])
gas = PILImage.create(get_image_files_sorted(path / 'train' / 'n03425413')[0])
church = church.resize((256, 256))
gas = gas.resize((256, 256))
tchurch = tensor(church).float() / 255.
tgas = tensor(gas).float() / 255.

_, axs = plt.subplots(1, 3, figsize=(12, 4))
show_image(tchurch, ax=axs[0])
show_image(tgas, ax=axs[1])
show_image((0.3 * tchurch + 0.7 * tgas), ax=axs[2])
