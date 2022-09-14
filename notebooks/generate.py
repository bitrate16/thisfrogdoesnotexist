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


	NUM_IMAGES     = 1024
	SAMPLES        = 8
	BEST_LIMIT_MAX = 1.0
	BEST_LIMIT_MIN = 0.0
	MAX_FAILS      = 50
	RESIZE_TO      = (64, 64)
	RESIZE_MODE    = Image.BICUBIC
	IMAGES_DIR     = 'generated-images'
	NET_D_PATH     = 'netD.pth'
	NET_G_PATH     = 'netG.pth'


	# Load models

	netD = torch.load(NET_D_PATH)
	netG = torch.load(NET_G_PATH)

	netD.eval()
	netG.eval()


	def pick_best_image(samples, best_limit_min, best_limit_max, max_fails, nz, device='cpu', resize_to=(64, 64), resize_sampling=Image.BICUBIC):
		with torch.no_grad():
			for m in range(max_fails):

				# Generate N images
				noise = torch.randn(samples, nz, 1, 1, device=device)
				fakes = netG(noise)

				# Select best for D metric
				output_fakes = netD(fakes)

				# Select best < best_limit
				masked_output_fakes = torch.masked_select(output_fakes, output_fakes.ge(best_limit_min))
				masked_output_fakes = torch.masked_select(masked_output_fakes, masked_output_fakes.le(best_limit_max))
				
				if len(masked_output_fakes) == 0:
					continue
				
				# Finx max with limit
				best_index = torch.argmax(masked_output_fakes)
				best = fakes[best_index]
				best_rate = masked_output_fakes[best_index]

				best_image = transforms.ToPILImage()(best * 0.5 + 0.5).convert('RGB').resize(resize_to, resample=resize_sampling)
				
				return best, best_image, best_rate
			
		return None, None, None


	ts = round(time.time())

	try:
		os.mkdir(IMAGES_DIR)
	except:
		pass
	try:
		os.mkdir(f'{IMAGES_DIR}/{ts}')
	except:
		pass

	for i in range(NUM_IMAGES):
		print(f'Image {i+1} of {NUM_IMAGES}')
		best, best_image, best_rate = pick_best_image(SAMPLES, BEST_LIMIT_MIN, BEST_LIMIT_MAX, MAX_FAILS, nz)
		best_image.resize(RESIZE_TO, resample=RESIZE_MODE).convert('RGB').save(f'{IMAGES_DIR}/{ts}/image_{i}.png')
