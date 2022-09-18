import os
import shutil
import tkinter as tk
from PIL import Image, ImageTk

INPUT = 'data-512'
OUTPUT = 'selected-data-512'

os.makedirs(OUTPUT, exist_ok=True)

window = tk.Tk()
window.geometry('600x650')

canvas = tk.Canvas(window, width=600, height=600)
canvas.pack()

files = os.listdir(INPUT)

# Resture states from output
states = [ 'reject' ] * len(files)
files_out = os.listdir(OUTPUT)
for f in files_out:
	try:
		index = files.index(f)
		states[index] = 'accept'
	except:
		pass
	
index = 0

image = ImageTk.PhotoImage(Image.open(f'{ INPUT }/{ files[0] }'))
image_container = canvas.create_image(0, 0, anchor='nw', image=image)

label = tk.Label(window, width=600, height=50)
label.place(relx=0, rely=600, anchor='nw')
label.config(text=f'[{ 0 } / { len(files) }] | { INPUT }/{ files[0] } : { states[0] }')
label.pack()

def set_image(new_index):
	global index, image
	index = (len(files) + (new_index % len(files))) % len(files) # max(0, min(new_index, len(files) - 1))
	
	print(f'{ INPUT }/{ files[index] }')
	image = ImageTk.PhotoImage(Image.open(f'{ INPUT }/{ files[index] }'))
	
	canvas.itemconfig(image_container, image=image)
	label.config(text=f'[{ index } / { len(files) }] | { INPUT }/{ files[index] } : { states[index] }')

def event_left(event):
	set_image(index - 1)

def event_right(event):
	set_image(index + 1)

def event_up(event):
	if states[index] == 'reject':
		states[index] = 'accept'
		label.config(text=f'{ INPUT }/{ files[index] } : { states[index] }')
		
		try:
			shutil.copy(f'{ INPUT }/{ files[index] }', f'{ OUTPUT }/{ files[index] }')
		except:
			pass

def event_down(event):
	if states[index] == 'accept':
		states[index] = 'reject'
		label.config(text=f'{ INPUT }/{ files[index] } : { states[index] }')
		
		try:
			os.remove(f'{ OUTPUT }/{ files[index] }')
		except:
			pass

window.bind('<Left>', event_left)
window.bind('<Right>', event_right)
window.bind('<Up>', event_up)
window.bind('<Down>', event_down)

window.mainloop()
