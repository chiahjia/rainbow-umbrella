import numpy as np
import cv2
import math
import os
from matplotlib import pyplot as plt


PATH = 'CFD/Images/' #path of the dataset, NOTE that i didnt include it in repo
folder = ['CFD-MR', 'CFD']

output_folder_path = 'gabor_output'
PI = math.pi

orientation = [0, PI/4, PI/2, 3*PI/4] #orientation of the filter in degree
freq = [0,2,4]

def load_downsam_data():

	folder_setup(output_folder_path)
	#used to calculate Gabor Kernal
	#kernel_list contains 12 filters
	kernel_list = calculateKernel()

	#loop through the image file in the dataset
	for temp_folder in folder:
		for img_name in os.listdir(PATH + temp_folder):
			img = cv2.imread(os.path.join(PATH+temp_folder, img_name), cv2.IMREAD_GRAYSCALE)

			#ignore some odd files in the dataset
			if 'Icon' in img_name:
				continue
			if '.DS_Store' in img_name:
				continue

			#get the type of expression of the image
			'''
			N: neutral
			A: angry
			F: fear
			HC: happy, closed mouth
			HO: happy, open mouth
			'''
			express_class = determine_out_location(img_name)

			#crop the image to 1280 x 960 and save it under gabor_output/[expression-type]_crop folder
			crop_img = img[378:1338, 581:1861]
			write_crop_img('crop_' + img_name, express_class, crop_img)
			
			#downsample the image from 1280 x 960 to 80 x 60 using pyramidImage
			#NOTE: a single cv2.pyrDown() call only reduce the # of pixels to max(1/2 x original image size)
			#      thus, multiple calls is required to downsample it to desire size
			ds_img = cv2.pyrDown(crop_img, dstsize=(int(crop_img.shape[1]/2), int(crop_img.shape[0]/2)))
			ds_img = cv2.pyrDown(ds_img, dstsize=(int(ds_img.shape[1]/2), int(ds_img.shape[0]/2)))
			ds_img = cv2.pyrDown(ds_img, dstsize=(int(ds_img.shape[1]/2), int(ds_img.shape[0]/2)))
			ds_img = cv2.pyrDown(ds_img, dstsize=(int(ds_img.shape[1]/2), int(ds_img.shape[0]/2)))

			output = []


			#apply the 12 filters on an image -> 12 resulting images from a single input
			#then, downsample the resulting image from 80x60 to 20x15
			#convert the resulting images to column vector of size 300x1
			#concatenate the 12 column vectors to form a 3600x1 vector
			#the 3600x1 vector is the final output for a single input image 
			#and is store in a [INPUT-IMAGE-NAME].txt file
			col_vec =[]
			i = 1
			for k in kernel_list:
				k_img = cv2.filter2D(ds_img, -1, k) #apply filter

				#downsample
				final_img = cv2.pyrDown(k_img, dstsize=(int(k_img.shape[1]/2), int(k_img.shape[0]/2)))
				final_img = cv2.pyrDown(final_img, dstsize=(int(final_img.shape[1]/2), int(final_img.shape[0]/2)))

				final_img_name = img_name[:len(img_name)-4] + '-filter'+ str(i) + '.jpg'
				final_img = final_img[:14, 5:15] #further crop the image for reduce the vector size

				#convert column vector
				if i == 1:
					col_vec = final_img.reshape(140, 1)
				else:
					temp_vec = final_img.reshape(140,1)
					col_vec = np.concatenate((col_vec, temp_vec), axis=0)
				
				write_img(final_img_name, express_class, final_img)
				i += 1

			write_txt(img_name[:len(img_name)-4], express_class, col_vec)

#function to calculate the Gabor kernel
#please refer the report for further explaination on this function
def calculateKernel():
	kernel_list = []
	for mu in orientation:
		for v in freq:
			mu_prime = mu  * PI / 8
			k_v = (2 ** (-(v + 2) / 2)) * 180

			wavelength  = ((k_v * math.cos(mu_prime))**2 + (k_v * math.sin(mu_prime))**2 ) ** (0.5)

			kernel = cv2.getGaborKernel((3,3), 2*PI, mu, wavelength, 0.5)
			kernel_list.append(kernel)

	return kernel_list

#function to help sort the result images
def determine_out_location(img_name):
	name_len = len(img_name)
	expression = img_name[name_len-5]

	return expression

#used to setup all the neccassary folders to store the result images
def folder_setup(output_path):

	if not os.path.exists(output_folder_path):
		os.mkdir(output_folder_path)

	if not os.path.exists(output_folder_path+'/N'):
		os.mkdir(output_folder_path+'/N')

	if not os.path.exists(output_folder_path+'/A'):
		os.mkdir(output_folder_path+'/A')

	if not os.path.exists(output_folder_path+'/F'):
		os.mkdir(output_folder_path+'/F')

	if not os.path.exists(output_folder_path+'/HC'):
		os.mkdir(output_folder_path+'/HC')

	if not os.path.exists(output_folder_path+'/HO'):
		os.mkdir(output_folder_path+'/HO')

	if not os.path.exists(output_folder_path+'/N_crop'):
		os.mkdir(output_folder_path+'/N_crop')

	if not os.path.exists(output_folder_path+'/A_crop'):
		os.mkdir(output_folder_path+'/A_crop')

	if not os.path.exists(output_folder_path+'/F_crop'):
		os.mkdir(output_folder_path+'/F_crop')

	if not os.path.exists(output_folder_path+'/HC_crop'):
		os.mkdir(output_folder_path+'/HC_crop')

	if not os.path.exists(output_folder_path+'/HO_crop'):
		os.mkdir(output_folder_path+'/HO_crop')

#funtion to create an image to different folders (according to their expression)
def write_img(img_name, express_class, img):

	if express_class == 'C':
		cv2.imwrite(output_folder_path + '/' + 'HC/' + img_name, img)
	elif express_class == 'O':
		cv2.imwrite(output_folder_path + '/' + 'HO/' + img_name, img)
	else:
		cv2.imwrite(output_folder_path + '/' + express_class + '/' + img_name, img)

#funtion to create an cropped image to different folders (according to their expression)
def write_crop_img(img_name, express_class, img):

	if express_class == 'C':
		cv2.imwrite(output_folder_path + '/' + 'HC_crop/' + img_name, img)
	elif express_class == 'O':
		cv2.imwrite(output_folder_path + '/' + 'HO_crop/' + img_name, img)
	else:
		cv2.imwrite(output_folder_path + '/' + express_class + '_crop/' + img_name, img)

#funtion to create a txt file to store the final vector and group them to different folders (according to their expression)
def write_txt(img_name, express_class, output_arr):
	if express_class == 'C':
		np.savetxt(output_folder_path + '/HC/' + img_name + '.txt' , output_arr, delimiter=',', fmt='%d')
	elif express_class == 'O':
		np.savetxt(output_folder_path + '/HO/' + img_name + '.txt' , output_arr, delimiter=',', fmt='%d')
	else:
		np.savetxt(output_folder_path + '/' + express_class + '/' + img_name + '.txt' , output_arr, delimiter=',', fmt='%d')


load_downsam_data()