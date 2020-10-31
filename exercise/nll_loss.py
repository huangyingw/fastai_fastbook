from fastai.callback.fp16 import *
from IPython.display import HTML
from fastai.vision.all import *
from fastbook import *
import fastbook
fastbook.setup_book()

acts = torch.randn((6, 2)) * 2
acts

targ = tensor([0, 1, 0, 1, 1, 0])

sm_acts = torch.softmax(acts, dim=1)
sm_acts

idx = range(6)

df = pd.DataFrame(sm_acts, columns=["3", "7"])
df['targ'] = targ
df['idx'] = idx
df['loss'] = sm_acts[range(6), targ]
t = df.style.hide_index()
# To have html code compatible with our script
html = t._repr_html_().split('</style>')[1]
html = re.sub(r'<table id="([^"]+)"\s*>', r'<table >', html)
display(HTML(html))

-sm_acts[idx, targ]

F.nll_loss(sm_acts, targ, reduction='none')

# ### Taking the Log

plot_function(torch.log, min=0, max=4)

loss_func = nn.CrossEntropyLoss()

loss_func(acts, targ)

F.cross_entropy(acts, targ)

nn.CrossEntropyLoss(reduction='none')(acts, targ)
