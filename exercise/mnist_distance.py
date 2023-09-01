from fastbook import *
from fastai.vision.all import *
import fastbook
fastbook.setup_book()

path = untar_data(URLs.MNIST_SAMPLE)

Path.BASE_PATH = path

path.ls()

(path / 'train').ls()

threes = (path / 'train' / '3').ls().sorted()
sevens = (path / 'train' / '7').ls().sorted()
threes

im3_path = threes[1]
im3 = Image.open(im3_path)
im3

array(im3)[4:10, 4:10]

tensor(im3)[4:10, 4:10]

im3_t = tensor(im3)
df = pd.DataFrame(im3_t[4:15, 4:22])
df.style.set_properties(**{'font-size': '6pt'}).background_gradient('Greys')

# ## First Try: Pixel Similarity

seven_tensors = [tensor(Image.open(o)) for o in sevens]
three_tensors = [tensor(Image.open(o)) for o in threes]
len(three_tensors), len(seven_tensors)

show_image(three_tensors[1])

stacked_sevens = torch.stack(seven_tensors).float() / 255
stacked_threes = torch.stack(three_tensors).float() / 255
stacked_threes.shape

len(stacked_threes.shape)

stacked_threes.ndim

mean3 = stacked_threes.mean(0)
show_image(mean3)

mean7 = stacked_sevens.mean(0)
show_image(mean7)

a_3 = stacked_threes[1]
show_image(a_3)

dist_3_abs = (a_3 - mean3).abs().mean()
dist_3_sqr = ((a_3 - mean3)**2).mean().sqrt()
dist_3_abs, dist_3_sqr

dist_7_abs = (a_3 - mean7).abs().mean()
dist_7_sqr = ((a_3 - mean7)**2).mean().sqrt()
dist_7_abs, dist_7_sqr

F.l1_loss(a_3.float(), mean7), F.mse_loss(a_3, mean7).sqrt()


valid_3_tens = torch.stack([tensor(Image.open(o))
                            for o in (path / 'valid' / '3').ls()])
valid_3_tens = valid_3_tens.float() / 255
valid_7_tens = torch.stack([tensor(Image.open(o))
                            for o in (path / 'valid' / '7').ls()])
valid_7_tens = valid_7_tens.float() / 255
valid_3_tens.shape, valid_7_tens.shape


def mnist_distance(a, b): return (a - b).abs().mean((-1, -2))


mnist_distance(a_3, mean3)

valid_3_dist = mnist_distance(valid_3_tens, mean3)
valid_3_dist, valid_3_dist.shape

tensor([1, 2, 3]) + tensor([1, 1, 1])

(valid_3_tens - mean3).shape


def is_3(x): return mnist_distance(x, mean3) < mnist_distance(x, mean7)


is_3(a_3), is_3(a_3).float()

is_3(valid_3_tens)

accuracy_3s = is_3(valid_3_tens).float() .mean()
accuracy_7s = (1 - is_3(valid_7_tens).float()).mean()

accuracy_3s, accuracy_7s, (accuracy_3s + accuracy_7s) / 2
