from __future__ import print_function

if __name__ == '__main__':
	
	import os
	import random
	import torch
	import torch.nn as nn
	import torch.nn.parallel
	import torch.optim as optim
	import torch.utils.data
	import torchvision.datasets as dset
	import torchvision.transforms as transforms
	import torchvision.utils as vutils
	from IPython.display import display
	from PIL import Image
	import time


	# Dataset location & load options
	dataroot = 'data-64'
	classes  = 'frogs'
	workers  = 8

	# Generator model location
	netG_path = None
	netD_path = None
	from_state = False

	# Output folder for snapshots
	outf = 'result'

	# Snapshot frequency (every $snap batches)
	model_snap = 50
	image_snap = 50

	# Snapshot frequency (every $snap_epoch epochs)
	model_snap_epoch = 1
	image_snap_epoch = 1

	# Specify what to snap:
	snap_state_dict = False
	snap_model = True

	# Cuda options
	cuda = True
	ngpu = 0

	# Size of input image (64 or 128)
	imageSize = 64

	# Number of channels
	nc = 3

	# Batch options
	batchSize = 128

	# Latent vector size
	nz = 10
	ngf = 64
	ndf = 64

	# Number of epochs
	niter = 50

	# Learning rate
	lr = 0.0001

	# ADAM: beta1
	beta1 = 0.5

	# Noise value
	noiseStd = 0.0
	noiseStdFinal = 0.0

	# Real labels range
	real_label_min = 1.0
	real_label_max = 1.0

	# Fake labels range
	fake_label_min = 0.0
	fake_label_max = 0.0

	# Percent of dropout for fake samples generator
	dropout_probability = 0.0

	# Generators seed
	seed = 101

	# Prepare for options
	try:
		os.makedirs(outf)
		os.makedirs(f'{outf}/models')
		os.makedirs(f'{outf}/states')
		os.makedirs(f'{outf}/images')
	except OSError:
		pass

	# Random for torch & others
	if seed is None:
		seed = random.randint(1, 10000)

	print("Random Seed: ", seed)
	random.seed(seed)
	torch.manual_seed(seed)

	# CUDA device select
	device = torch.device("cuda:0" if cuda else "cpu")


	dataset = dset.ImageFolder(root=dataroot,
							transform=transforms.Compose([
								transforms.Resize(imageSize),
								transforms.CenterCrop(imageSize),
								transforms.ToTensor(),
								transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
							]))

	assert dataset
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=workers)


	def weights_init(m):
		classname = m.__class__.__name__
		if classname.find('Conv') != -1:
			torch.nn.init.normal_(m.weight, 0.0, 0.02)
		elif classname.find('BatchNorm') != -1:
			torch.nn.init.normal_(m.weight, 1.0, 0.02)
			torch.nn.init.zeros_(m.bias)

	class Generator(nn.Module):
		def __init__(self, ngpu):
			super(Generator, self).__init__()
			self.ngpu = ngpu
			if imageSize == 64:
				self.main = nn.Sequential(
					# input is Z, going into a convolution
					nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
					nn.BatchNorm2d(ngf * 8),
					nn.ReLU(True),
					nn.Dropout(dropout_probability),
					# state size. (ngf*8) x 4 x 4
					nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
					nn.BatchNorm2d(ngf * 4),
					nn.ReLU(True),
					nn.Dropout(dropout_probability),
					# state size. (ngf*4) x 8 x 8
					nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
					nn.BatchNorm2d(ngf * 2),
					nn.ReLU(True),
					nn.Dropout(dropout_probability),
					# state size. (ngf*2) x 16 x 16
					nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
					nn.BatchNorm2d(ngf),
					nn.ReLU(True),
					nn.Dropout(dropout_probability),
					# state size. (ngf) x 32 x 32
					nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
					nn.Tanh(),
					nn.Dropout(dropout_probability)
					# state size. (nc) x 64 x 64
				)
			elif imageSize == 128:
				self.main = nn.Sequential(
					# input is Z, going into a convolution
					nn.ConvTranspose2d(     nz, ngf * 16, 4, 1, 0, bias=False),
					nn.BatchNorm2d(ngf * 16),
					nn.ReLU(True),
					nn.Dropout(dropout_probability),
					# state size. (ngf*16) x 4 x 4
					nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
					nn.BatchNorm2d(ngf * 8),
					nn.ReLU(True),
					nn.Dropout(dropout_probability),
					# state size. (ngf*8) x 8 x 8
					nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
					nn.BatchNorm2d(ngf * 4),
					nn.ReLU(True),
					nn.Dropout(dropout_probability),
					# state size. (ngf*4) x 16 x 16 
					nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
					nn.BatchNorm2d(ngf * 2),
					nn.ReLU(True),
					nn.Dropout(dropout_probability),
					# state size. (ngf*2) x 32 x 32
					nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
					nn.BatchNorm2d(ngf),
					nn.ReLU(True),
					nn.Dropout(dropout_probability),
					# state size. (ngf) x 64 x 64
					nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
					nn.Tanh(),
					nn.Dropout(dropout_probability)
					# state size. (nc) x 128 x 128
				)

		def forward(self, input):
			if input.is_cuda and self.ngpu > 1:
				output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
			else:
				output = self.main(input)
			return output
		
	class Discriminator(nn.Module):
		def __init__(self, ngpu):
			super(Discriminator, self).__init__()
			self.ngpu = ngpu
			if imageSize == 64:
				self.main = nn.Sequential(
					# input is (nc) x 64 x 64
					nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
					nn.LeakyReLU(0.2, inplace=True),
					# state size. (ndf) x 32 x 32
					nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
					nn.BatchNorm2d(ndf * 2),
					nn.LeakyReLU(0.2, inplace=True),
					# state size. (ndf*2) x 16 x 16
					nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
					nn.BatchNorm2d(ndf * 4),
					nn.LeakyReLU(0.2, inplace=True),
					# state size. (ndf*4) x 8 x 8
					nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
					nn.BatchNorm2d(ndf * 8),
					nn.LeakyReLU(0.2, inplace=True),
					# state size. (ndf*8) x 4 x 4
					nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
					nn.Sigmoid()
					# state size. 1
				)
			elif imageSize == 128:
				self.main = nn.Sequential(
					# input is (nc) x 128 x 128
					nn.Conv2d(nc, ndf, 4, stride=2, padding=1, bias=False), 
					nn.LeakyReLU(0.2, inplace=True),
					# state size. (ndf) x 64 x 64
					nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False),
					nn.BatchNorm2d(ndf * 2),
					nn.LeakyReLU(0.2, inplace=True),
					# state size. (ndf*2) x 32 x 32
					nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
					nn.BatchNorm2d(ndf * 4),
					nn.LeakyReLU(0.2, inplace=True),
					# state size. (ndf*4) x 16 x 16 
					nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False),
					nn.BatchNorm2d(ndf * 8),
					nn.LeakyReLU(0.2, inplace=True),
					# state size. (ndf*8) x 8 x 8
					nn.Conv2d(ndf * 8, ndf * 16, 4, stride=2, padding=1, bias=False),
					nn.BatchNorm2d(ndf * 16),
					nn.LeakyReLU(0.2, inplace=True),
					# state size. (ndf*16) x 4 x 4
					nn.Conv2d(ndf * 16, 1, 4, stride=1, padding=0, bias=False),
					nn.Sigmoid()
					# state size. 1
				)

		def forward(self, input):
			if input.is_cuda and self.ngpu > 1:
				output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
			else:
				output = self.main(input)

			return output.view(-1, 1).squeeze(1)

	if from_state:
		netG = Generator(ngpu)
		netG.apply(weights_init)
		if netG_path is not None:
			netG.load_state_dict(torch.load(netG_path))
		#print(netG)

		netD = Discriminator(ngpu)
		netD.apply(weights_init)
		if netD_path is not None:
			netD.load_state_dict(torch.load(netD_path))
		#print(netD)
	else:
		if netG_path is not None:
			netG = torch.load(netG_path)
		else:
			netG = Generator(ngpu)
			netG.apply(weights_init)
		#print(netG)
		
		if netD_path is not None:
			netD = torch.load(netD_path)
		else:
			netD = Discriminator(ngpu)
			netD.apply(weights_init)
		#print(netD)


	criterion = nn.BCELoss()
	fixed_noise = torch.randn(batchSize, nz, 1, 1, device=device)

	# torch.autograd.set_detect_anomaly(True)

	# setup optimizer
	optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999)) # SGD(netD.parameters(), lr=lr)
	optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
	
	netG.to(device)
	netD.to(device)
	netG.train()
	netD.train()

	for epoch in range(niter):
		# Decay noise depending on iterations count
		noiseStdCurrent = noiseStd + (noiseStdFinal - noiseStd) / (niter - epoch)

		for i, data in enumerate(dataloader, 0):
			############################
			# (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
			###########################

			# train with real
			netD.zero_grad()
			real_cpu = data[0].to(device)

			# Add noise
			real_cpu = real_cpu + torch.randn(real_cpu.size(), device=device) * noiseStdCurrent

			batch_size = real_cpu.size(0)

			# Generate label distribution
			real_label = torch.rand((batch_size,), dtype=real_cpu.dtype, device=device) * (real_label_max - real_label_min) + real_label_min # torch.full((batch_size,), real_label, dtype=real_cpu.dtype, device=device)

			# Flip labels with chance of 0.001
			# ?

			output = netD(real_cpu)
			errD_real = criterion(output, real_label) ###############################################################
			errD_real.backward()
			D_x = output.mean().item()

			# train with fake
			noise = torch.randn(batch_size, nz, 1, 1, device=device)
			fake = netG(noise)

			# Generate label distribution
			fake_label = torch.rand((batch_size,), dtype=real_cpu.dtype, device=device) * (fake_label_max - fake_label_min) + fake_label_min # .fill_(fake_label)

			output = netD(fake.detach())
			errD_fake = criterion(output, fake_label)
			errD_fake.backward()
			D_G_z1 = output.mean().item()
			errD = errD_real + errD_fake
			optimizerD.step()

			############################
			# (2) Update G network: maximize log(D(G(z)))
			###########################
			netG.zero_grad()
			output = netD(fake)
			errG = criterion(output, real_label)
			errG.backward()
			D_G_z2 = output.mean().item()
			optimizerG.step()

			print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % (epoch, niter, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
			
			# Snap batches
			if i % model_snap == 0:
				if snap_model:
					torch.save(netG, f'{outf}/models/netG_res_{imageSize}_seed_{seed}_epoch_{epoch}_iter_{i}.pth')
					torch.save(netD, f'{outf}/models/netD_res_{imageSize}_seed_{seed}_epoch_{epoch}_iter_{i}.pth')
				if snap_state_dict:
					torch.save(netG.state_dict(), f'{outf}/states/netG_res_{imageSize}_seed_{seed}_epoch_{epoch}_iter_{i}.pth')
					torch.save(netD.state_dict(), f'{outf}/states/netD_res_{imageSize}_seed_{seed}_epoch_{epoch}_iter_{i}.pth')
			
			if i % image_snap == 0:
				netG.eval()
				netD.eval()
				vutils.save_image(real_cpu, f'{outf}/images/res_{imageSize}_seed_{seed}.png', normalize=True)
				fake = netG(fixed_noise).cpu()
				vutils.save_image(fake.detach(), f'{outf}/images/res_{imageSize}_seed_{seed}_epoch_{epoch}_iter_{i}.png', normalize=True)
				netG.train()
				netD.train()
		
		# Snap epochs
		if epoch % model_snap_epoch == 0:
			if snap_model:
				torch.save(netG, f'{outf}/models/netG_res_{imageSize}_seed_{seed}_epoch_{epoch}_final.pth')
				torch.save(netD, f'{outf}/models/netD_res_{imageSize}_seed_{seed}_epoch_{epoch}_final.pth')
			if snap_state_dict:
				torch.save(netG.state_dict(), f'{outf}/states/netG_res_{imageSize}_seed_{seed}_epoch_{epoch}_final.pth')
				torch.save(netD.state_dict(), f'{outf}/states/netD_res_{imageSize}_seed_{seed}_epoch_{epoch}_final.pth')
		
		if epoch % image_snap_epoch == 0:
			netG.eval()
			netD.eval()
			vutils.save_image(real_cpu, f'{outf}/images/res_{imageSize}_seed_{seed}.png', normalize=True)
			fake = netG(fixed_noise).cpu()
			vutils.save_image(fake.detach(), f'{outf}/images/res_{imageSize}_seed_{seed}_epoch_{epoch}_final.png', normalize=True)
			netG.train()
			netD.train()
		
	# Snap last
	if snap_model:
		torch.save(netG, f'{outf}/models/netG_res_{imageSize}_seed_{seed}_final.pth')
		torch.save(netD, f'{outf}/models/netD_res_{imageSize}_seed_{seed}_final.pth')
	if snap_state_dict:
		torch.save(netG.state_dict(), f'{outf}/states/netG_res_{imageSize}_seed_{seed}_final.pth')
		torch.save(netD.state_dict(), f'{outf}/states/netD_res_{imageSize}_seed_{seed}_final.pth')

	netG.eval()
	netD.eval()
	vutils.save_image(real_cpu, f'{outf}/images/res_{imageSize}_seed_{seed}.png', normalize=True)
	fake = netG(fixed_noise).cpu()
	vutils.save_image(fake.detach(), f'{outf}/images/res_{imageSize}_seed_{seed}_final.png', normalize=True)
