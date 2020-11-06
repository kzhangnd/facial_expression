import os
import imageio
from os import path, makedirs
import cv2
from tqdm import tqdm 


root = 'yalefaces'
dest = 'yalefaces_curated'
for f in tqdm(os.listdir(root)):
	file_path = path.join(root, f)
	if f.endswith('.txt') or f[0] == '.':	# if the file is Readme.txt or .DS_store
		continue
	if not path.isfile(file_path):
		continue

	img = imageio.mimread(file_path)[0]
	img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # convert from rgb to bgr


	subject = f.split('.')[0]
	attribute = f.split('.')[1]
	# create a folder for each subject
	subject_folder = path.join(dest, subject)
	if not path.exists(subject_folder):
		makedirs(subject_folder)
	
	
	destination = path.join(subject_folder, attribute+'.jpg')
	
	cv2.imwrite(destination, img_bgr) # save the image using cv2

	