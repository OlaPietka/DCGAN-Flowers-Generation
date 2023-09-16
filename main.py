from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

import configs as c
from dataset import get_transform, get_dataloader
from models import Generator, Discriminator
from train import DCGAN
from utils import get_noise, weights_init


if __name__ == "__main__":
	transform = get_transform((c.img_size, c.img_size))
	dataloader = get_dataloader(c.data_dir, transform, c.batch_size)

	g = Generator(c.img_channels, c.z_dim, c.g_hidden_layer).to(c.device)
	g_opt = Adam(g.parameters(), lr=c.g_lr, betas=(c.beta_1, c.beta_2))

	d = Discriminator(c.img_channels, c.d_hidden_layer).to(c.device)
	d_opt = Adam(d.parameters(), lr=c.d_lr, betas=(c.beta_1, c.beta_2))

	g_scheduler = StepLR(g_opt, step_size=c.step_size, gamma=c.gamma)
	d_scheduler = StepLR(d_opt, step_size=c.step_size, gamma=c.gamma)

	g = g.apply(weights_init)
	d = d.apply(weights_init)

	fixed_noise = get_noise(c.batch_size, c.z_dim, device=c.device)

	dcgan = DCGAN(
		generator=g,
		discriminator=d,
		g_optimizer=g_opt,
		d_optimizer=d_opt,
		g_scheduler=g_scheduler,
		d_scheduler=d_scheduler,
		criterion=c.criterion,
		z_dim=c.z_dim,
		epochs=c.epochs,
		dataloader=dataloader,
		fixed_noise=fixed_noise,
		device=c.device,
	)
	dcgan.train(display_step=c.display_step, save_dir=c.save_dir)
