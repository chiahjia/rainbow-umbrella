import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from config import cfg


def demo_nmf(filenames):
    base = cv2.cvtColor(cv2.imread(filenames[0]), cv2.COLOR_BGR2GRAY)
    test = cv2.cvtColor(cv2.imread(filenames[1]), cv2.COLOR_BGR2GRAY)

    model = NMF(n_components=10, init='random', random_state=0, max_iter=1000)

    W = model.fit_transform(base)
    H = model.components_
    W2 = model.transform(test)
    test_after = np.matmul(W2, H)
    base_after = np.matmul(W, H)

    plt.subplot(2, 2, 1)
    plt.imshow(base, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.imshow(base_after, cmap='gray')

    plt.subplot(2, 2, 3)
    plt.imshow(test, cmap='gray')
    plt.subplot(2, 2, 4)
    plt.imshow(test_after, cmap='gray')

    plt.show()

# expression code to directory name mapping
expr_dict = dict(
    N='neutral',
    A='angry',
    F='fear',
    HC='happy_closed',
    HO='happy_open',
)

def reduce_all_imgs():
    source = cfg['source_folder']
    dest = cfg['dest_folder_nmf']
    img_extensions = cfg['valid_img_extensions']

    # build dest folders if required
    for _, expr_dir in expr_dict.items():
        expr_path = os.path.join(dest, expr_dir)
        try: os.mkdir(expr_path)
        except: pass  # exception occurs if 

    model = get_model(source)
    cwd = os.listdir(source)

    # iterate over directories in source dir
    for path in cwd:
        if not os.path.isdir(path): continue
        
        # iterate over imgs in this subject's dir
        os.chdir(path)
        for path2 in os.listdir(os.getcwd()):
            if not os.path.isfile(path2): continue
            
            # confirm that it's an img file
            i_dot = path2.rfind('.')
            if i_dot == -1: continue
            ext = path2[i_dot+1:]
            if ext not in img_extensions: continue

            # get expression code
            i_dash = path2.rfind('-')
            expr_code = path2[i_dash+1:i_dot]

            # transform img and save to dest folder
            img = cv2.cv2.cvtColor(cv2.imread(path2), cv2.COLOR_BGR2GRAY)
            img = crop_and_downsample(img)
            W = model.transform(img)

            # save to dest folder
            os.chdir(dest)
            os.chdir(expr_dict[expr_code])
            np.savetxt(f'{path2[:i_dot]}.txt', W)

            # plotting for testing purposes
            img_after = np.matmul(W, model.components_)
            plt.subplot(1, 2, 1)
            plt.imshow(img, cmap='gray')
            plt.subplot(1, 2, 2)
            plt.imshow(img_after, cmap='gray')

            plt.show()
            exit()

        os.chdir(source)

def crop_and_downsample(img):
    crop_img = img[378:1338, 581:1861]
    
    #downsample the image from 1280 x 960 to 80 x 60 using pyramidImage
    #NOTE: a single cv2.pyrDown() call only reduce the # of pixels to max(1/2 x original image size)
    #      thus, multiple calls is required to downsample it to desire size
    ds_img = cv2.pyrDown(crop_img, dstsize=(int(crop_img.shape[1]/2), int(crop_img.shape[0]/2)))
    ds_img = cv2.pyrDown(ds_img, dstsize=(int(ds_img.shape[1]/2), int(ds_img.shape[0]/2)))
    ds_img = cv2.pyrDown(ds_img, dstsize=(int(ds_img.shape[1]/2), int(ds_img.shape[0]/2)))
    ds_img = cv2.pyrDown(ds_img, dstsize=(int(ds_img.shape[1]/2), int(ds_img.shape[0]/2)))
    return ds_img


def get_model(source):
    os.chdir(source)
    foldername = cfg['nmf_basis_img_folder']
    filename = cfg['nmf_basis_img_file']
    os.chdir(foldername)
    
    basis_img = cv2.cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)
    basis_img = crop_and_downsample(basis_img)

    dim = cfg['nmf_dim']
    max_iter = cfg['nmf_max_iter']

    model = NMF(n_components=dim, init='random', random_state=0, max_iter=max_iter)
    model.fit(basis_img)

    os.chdir(source)
    return model


if __name__ == '__main__':
    # if running demo, update filenames first
    # filenames = ['people2.jpg', 'people3.jpg']
    # demo_nmf()

    reduce_all_imgs()