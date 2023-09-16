from torch import nn

img_channels = 3
img_size = 128
img_shape = (img_channels, img_size, img_size)

criterion = nn.BCELoss()
epochs = 500
batch_size = 128

z_dim = 300
g_hidden_layer = 64
d_hidden_layer = 64

display_step = 100

g_lr = 0.0005
d_lr = 0.0005

step_size = 50
gamma = 0.9

beta_1 = 0.5
beta_2 = 0.999

device = "cuda"  # either 'cpu' or 'cuda'
data_dir = None  # 'path/to/data/dir/'
save_dir = None  # 'path/to/save/dir/'
