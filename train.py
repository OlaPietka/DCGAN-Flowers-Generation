import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import Discriminator, Generator
from utils import save_tensor_images, get_noise


class DCGAN:
	def __init__(
			self,
			generator: Generator,
			discriminator: Discriminator,
			g_optimizer: Optimizer,
			d_optimizer: Optimizer,
			criterion: Module,
			z_dim: int,
			epochs: int,
			dataloader: DataLoader,
			fixed_noise: Tensor,
			g_scheduler: StepLR,
			d_scheduler: StepLR,
			device: str ="cuda",
	):
		self.G = generator
		self.D = discriminator
		self.G_opt = g_optimizer
		self.D_opt = d_optimizer
		self.criterion = criterion
		self.z_dim = z_dim
		self.epochs = epochs
		self.dataloader = dataloader
		self.fixed_noise = fixed_noise
		self.G_scheduler = g_scheduler
		self.D_scheduler = d_scheduler
		self.device = device

		self.G_losses = []
		self.D_losses = []

	def _update_discriminator(self, fake: Tensor, real: Tensor) -> None:
		self.D_opt.zero_grad()

		fake_pred = self.D(fake.detach())
		fake_loss = self.criterion(fake_pred, self._zeros_like(fake_pred))
		real_pred = self.D(real)
		real_loss = self.criterion(real_pred, self._ones_like(real_pred))

		D_loss = (fake_loss + real_loss) / 2
		self.D_losses.append(D_loss.item())
		D_loss.backward()

		self.D_opt.step()

	def _update_generator(self, fake: Tensor) -> None:
		self.G_opt.zero_grad()

		fake_pred = self.D(fake)

		G_loss = self.criterion(fake_pred, self._ones_like(fake_pred))
		self.G_losses.append(G_loss.item())
		G_loss.backward()

		self.G_opt.step()

	def _update_schedulers(self) -> None:
		self.D_scheduler.step()
		self.G_scheduler.step()

	def _ones_like(self, x: Tensor) -> Tensor:
		return torch.ones_like(x).to(self.device)

	def _zeros_like(self, x: Tensor) -> Tensor:
		return torch.zeros_like(x).to(self.device)

	def _display_info(self, epoch: int, display_step: int, current_step: int, save_dir: str) -> None:
		G_mean = sum(self.G_losses[-display_step:]) / display_step
		D_mean = sum(self.D_losses[-display_step:]) / display_step

		print(f"[epoch {epoch}, step {current_step}]: g_loss = {G_mean:.8f}, d_loss = {D_mean:.8f}")

		fake = self.G(self.fixed_noise)
		save_tensor_images(fake.detach(), save_dir + f"{current_step}.jpg")

	def _save(self, save_dir: str) -> None:
		torch.save(self.G.state_dict(), save_dir + "generator.pt")
		torch.save(self.D.state_dict(), save_dir + "discriminator.pt")

	def train(self, save_dir: str, display_step: int = 100) -> None:
		current_step = 0

		for epoch in range(self.epochs):
			for real in tqdm(self.dataloader):
				noise = get_noise(len(real), self.z_dim, device=self.device)

				fake = self.G(noise)

				self._update_discriminator(fake, real.to(self.device))
				self._update_generator(fake)

				if current_step % display_step == 0 and current_step > 0:
					self._display_info(epoch, display_step, current_step, save_dir)

				current_step += 1

			self._update_schedulers()

		self._save(save_dir)
