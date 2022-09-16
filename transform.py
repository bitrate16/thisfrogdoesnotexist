# This file is used for augmenting input dataset statically.
# Instead, PyTorch transforms could be used in pair with random rorations.

from PIL import Image, ImageOps
from multiprocessing import Pool
import os
import colorsys

# Number of workers
NUM_WORKERS = 8

# Path to images
IMAGES_PATHS = [ 'data-256/frogs' ]

# Define allowed transformations
ROTATIONS = [ -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45 ]
HSV_SHIFTS = [ 0 ]

# Output size (size x size)
OUTPUT_SIZE = 128
OUTPUT_RESAMPLE = Image.BICUBIC
OUTPUT = 'tdata-128/frogs'

# Delete all transformed images, but keep originals
CLEANUP = False

###########################

# https://stackoverflow.com/a/24875271
def shift_hsv(image, angle):
	im = image.copy()
	im = im.convert('RGB')
	ld = im.load()
	width, height = im.size
	for y in range(height):
		for x in range(width):
			r, g, b = ld[x, y]
			h, s, v = colorsys.rgb_to_hsv(r / 255., g / 255., b / 255.)
			h = (h + -angle / 360.0) % 1.0
			s = s ** 0.65
			r, g, b = colorsys.hsv_to_rgb(h, s, v)
			ld[x, y] = (int(r * 255.9999), int(g * 255.9999), int(b * 255.9999))
	
	im = im.convert('RGBA')
	return im

# Flip horisontal
def mirror_image(image):
	image = image.copy()
	im_mirror = ImageOps.mirror(image)
	return im_mirror

# Rotate angle
def rotate_image(image, angle):
	im_rot = image.rotate(angle)
	bg = Image.new('RGBA', image.size, (255, 255, 255, 255))
	bg.paste(im_rot, (0, 0), im_rot)
	return bg

# Process single image file
def process_image(args):
	file, dstfile = args
	print(f'Transform {file}')
		
	no_ext_file = dstfile.split('.png')[0]

	if '-transformed' in file:
		if CLEANUP:
			print(f'Delete {file}')
			os.remove(file)
		return
	
	if CLEANUP:
		return

	try:
		image = Image.open(file).convert('RGBA')
		image.convert('RGB').save(file)

		for flip in [ False, True ]:
			image_flip = mirror_image(image) if flip else image
			
			for rot in ROTATIONS:
				image_rot = image_flip.copy() if rot == 0 else rotate_image(image_flip, rot)

				for hsv in HSV_SHIFTS:
					# # No save because equal to original
					# if (not flip) and (rot == 0) and (hsv == 0):
					# 	continue
					
						image_hsv = image_rot.copy() if hsv == 0 else shift_hsv(image_rot, hsv)					
						image_hsv = image_hsv.convert('RGB').resize((OUTPUT_SIZE, OUTPUT_SIZE), resample=OUTPUT_RESAMPLE)
						image_hsv.save(f'{no_ext_file}_{flip}_{rot}_{hsv}-transformed.png')
	except:
		print(f'FAIL: {file}')


if __name__ == '__main__':
	# Collect all required files
	files = []
	for IMAGES_PATH in IMAGES_PATHS:
		dir_files = list(os.listdir(IMAGES_PATH))
		for file in dir_files:
			files.append((f'{IMAGES_PATH}/{file}', f'{OUTPUT}/{file}'))
	
	os.makedirs(OUTPUT, exist_ok=True)
	
	pool = Pool(NUM_WORKERS)
	pool.map(process_image, files)