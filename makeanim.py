import os
import shutil
import subprocess

from multiprocessing.pool import ThreadPool


BASE_PATH = 'images'
TEMP_PATH = 'C:/temp/temp_gif'
THREADS = os.cpu_count()
COPY_ONSTART = False
DELETE_ATEXIT = False


def extractor(v: str):
	v = v[:-4]
	if '/' in v:
		v = v[v.rindex('/'):]
	p = v.split('_')
	return int(p[5]) * 1000000000 + int(p[7])

def copy(f):
	print(f[1])
	shutil.copy(f'{ BASE_PATH }/{ f[1] }', f'{ TEMP_PATH }/{ f[0] }.png')


files = [ f for f in os.listdir(BASE_PATH) if 'iter' in f ]
files.sort(key=extractor)


if COPY_ONSTART:
	print('Copying files')
	os.makedirs(TEMP_PATH, exist_ok=True)
	with ThreadPool(THREADS) as pool:
		pool.map(copy, enumerate(files))


print('Starting ffmpeg')
process = subprocess.Popen(f'ffmpeg -i "{ TEMP_PATH }/%d.png" out.mp4', shell=True, stdout=subprocess.PIPE)
for line in process.stdout:
	print(line)
process.wait()


if DELETE_ATEXIT:
	print('Deleting files')
	shutil.rmtree(TEMP_PATH)
