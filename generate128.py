


if __name__ == '__main__':
	
	import os
	import random
	import argparse
	import torch
	import torch.nn.parallel
	import torch.nn as nn
	import torch.utils.data
	import torchvision.utils as vutils
	from tqdm import tqdm
	
	# Configure
	nc = 3
	nz = 100
	ngf = 32
	ndf = 32


	class Generator(nn.Module):
		def __init__(self, ngpu):
			super(Generator, self).__init__()
			self.ngpu = ngpu
			self.main = nn.Sequential(
				# input is Z, going into a convolution
				nn.ConvTranspose2d(     nz, ngf * 16, 4, 1, 0, bias=False),
				nn.BatchNorm2d(ngf * 16),
				nn.ReLU(True),
				# state size. (ngf*16) x 4 x 4
				nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
				nn.BatchNorm2d(ngf * 8),
				nn.ReLU(True),
				# state size. (ngf*8) x 8 x 8
				nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
				nn.BatchNorm2d(ngf * 4),
				nn.ReLU(True),
				# state size. (ngf*4) x 16 x 16 
				nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
				nn.BatchNorm2d(ngf * 2),
				nn.ReLU(True),
				# state size. (ngf*2) x 32 x 32
				nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
				nn.BatchNorm2d(ngf),
				nn.ReLU(True),
				# state size. (ngf) x 64 x 64
				nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
				nn.Tanh()
				# state size. (nc) x 128 x 128
			)
		def forward(self, input):
			return self.main(input)


	def positive_count(x):
		x = int(x)
		if x < 0:
			raise argparse.ArgumentTypeError("Minimum count is 1")
		return x

	parser = argparse.ArgumentParser()
	parser.add_argument('--seed', default=42, type=int)
	parser.add_argument('--count', default=1, type=positive_count)
	parser.add_argument('--output', default='images', type=str)
	parser.add_argument('--netG', default='netG.pth', type=str)
	parser.add_argument('--cuda', action='store_true')
	args = parser.parse_args()
	
	random.seed(args.seed)
	torch.manual_seed(args.seed)
	
	try:
		os.makedirs(args.output)
	except:
		pass
	
	device = 'cuda:0' if args.cuda else 'cpu'
	
	netG = torch.load(args.netG)
	netG.to(device)
	netG.eval()
	
	for i in tqdm(range(args.count)):
		noise = torch.randn(1, nz, 1, 1, device=device)
		output = netG(noise).cpu()
		vutils.save_image(output.detach(), f'{ args.output }/image_{ i :06d}.png', normalize=True)
