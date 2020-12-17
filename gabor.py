import numpy as np
import cv2
import math
import os
from matplotlib import pyplot as plt


PATH = 'CFD/Images/'
folder = ['CFD-MR', 'CFD']
ethnic_gen = ['WM', 'WF', 'LM', 'LF', 'BM', 'BF', 'AM', 'AF']

output_folder_path = 'gabor_output'


PI = math.pi

orientation = [0, PI/4, PI/2, 3*PI/4] #orientation of the filter in degree
freq = [0,2,4]

def load_downsam_data():

	folder_setup(output_folder_path)

	for img_name in os.listdir(PATH + folder[0]):
		img = cv2.imread(os.path.join(PATH+folder[0], img_name), cv2.IMREAD_GRAYSCALE)

		print(img_name)
		if 'Icon' in img_name:
			continue
		if '.DS_Store' in img_name:
			continue

		express_class = determine_out_location(img_name)

		crop_img = img[378:1338, 581:1861]
		write_crop_img('crop_' + img_name, express_class, crop_img)
		print(crop_img.shape)
		
		ds_img = cv2.pyrDown(crop_img, dstsize=(int(crop_img.shape[1]/2), int(crop_img.shape[0]/2)))
		ds_img = cv2.pyrDown(ds_img, dstsize=(int(ds_img.shape[1]/2), int(ds_img.shape[0]/2)))
		ds_img = cv2.pyrDown(ds_img, dstsize=(int(ds_img.shape[1]/2), int(ds_img.shape[0]/2)))
		ds_img = cv2.pyrDown(ds_img, dstsize=(int(ds_img.shape[1]/2), int(ds_img.shape[0]/2)))

		#print(ds_img.shape)
		#plt.imshow(ds_img, cmap='gray')
		#plt.show()

		output = []
		kernel_list = calculateKernel()

		col_vec =[]
		i = 1
		for k in kernel_list:
			k_img = cv2.filter2D(ds_img, -1, k)

			final_img = cv2.pyrDown(k_img, dstsize=(int(k_img.shape[1]/2), int(k_img.shape[0]/2)))
			final_img = cv2.pyrDown(final_img, dstsize=(int(final_img.shape[1]/2), int(final_img.shape[0]/2)))
			#plt.subplot(4,3,i)
			#plt.imshow(final_img, cmap='gray')
			final_img_name = img_name[:len(img_name)-4] + '-filter'+ str(i) + '.jpg'

			print(final_img)
			if i == 1:
				col_vec = final_img.reshape(300, 1)
				#print(col_vec)
			else:
				temp_vec = final_img.reshape(300,1)
				col_vec = np.concatenate((col_vec, temp_vec), axis=0)

			print('col_vec shape is ' + str(col_vec.shape))
			#f = open(output_folder_path + "/test.txt", "w")
			#f.write(str(col_vec))
			#np.savetxt(output_folder_path + '/'+ "/test.txt", col_vec, delimiter=',', fmt='%d')
			
			write_img(final_img_name, express_class, final_img)
			i += 1
		write_txt(img_name[:len(img_name)-4], express_class, col_vec)


		#plt.show()
		
		

def calculateKernel():
	kernel_list = []
	for mu in orientation:
		for v in freq:
			mu_prime = mu  * PI / 8
			k_v = (2 ** (-(v + 2) / 2)) * 180

			wavelength  = ((k_v * math.cos(mu_prime))**2 + (k_v * math.sin(mu_prime))**2 ) ** (0.5)
			#print(wavelength)

			kernel = cv2.getGaborKernel((3,3), 2*PI, mu, wavelength, 0.5)
			kernel_list.append(kernel)

	return kernel_list

def determine_out_location(img_name):
	name_len = len(img_name)
	expression = img_name[name_len-5]

	return expression

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

def write_img(img_name, express_class, img):

	if express_class == 'C':
		cv2.imwrite(output_folder_path + '/' + 'HC/' + img_name, img)
	elif express_class == 'O':
		cv2.imwrite(output_folder_path + '/' + 'HO/' + img_name, img)
	else:
		cv2.imwrite(output_folder_path + '/' + express_class + '/' + img_name, img)

def write_crop_img(img_name, express_class, img):

	if express_class == 'C':
		cv2.imwrite(output_folder_path + '/' + 'HC_crop/' + img_name, img)
	elif express_class == 'O':
		cv2.imwrite(output_folder_path + '/' + 'HO_crop/' + img_name, img)
	else:
		cv2.imwrite(output_folder_path + '/' + express_class + '_crop/' + img_name, img)

def write_txt(img_name, express_class, output_arr):
	if express_class == 'C':
		np.savetxt(output_folder_path + '/HC/' + img_name + '.txt' , output_arr, delimiter=',', fmt='%d')
	elif express_class == 'O':
		np.savetxt(output_folder_path + '/HO/' + img_name + '.txt' , output_arr, delimiter=',', fmt='%d')
	else:
		np.savetxt(output_folder_path + '/' + express_class + '/' + img_name + '.txt' , output_arr, delimiter=',', fmt='%d')


load_downsam_data()