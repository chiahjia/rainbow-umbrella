import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from config import cfg


def estimate_mouth(img, occlusion_type, files_dir=None, show=False):
    if not files_dir:
        files_dir = os.getcwd()

    cascade = cv2.CascadeClassifier(os.path.join(files_dir, 'haarcascade_frontalface_default.xml'))
    eye2_cascade = cv2.CascadeClassifier(os.path.join(files_dir, 'haarcascade_eye_tree_eyeglasses.xml'))

    faces = cascade.detectMultiScale(img, 1.1, 4)
    eyes = eye2_cascade.detectMultiScale(img, 1.1, 4)

    if len(faces) == 0 or len(eyes) == 0:
        return None

    face = faces[0]

    # occlude face based on selected method
    if occlusion_type == FM:
        mouth = get_full_mask_occlusion(face, eyes)
    elif occlusion_type == MO:
        mouth = get_small_occlusion(face, eyes)
    else:
        print('Unknown/unimplemented occlusion type')
        exit()

    mouth_x, mouth_y, mouth_w, mouth_h = mouth

    if show:
        img[mouth_y:mouth_y+mouth_h, mouth_x:mouth_x+mouth_w] = 0
        cv2.imshow("ting", cv2.resize(img, (1000, 1000)))
        cv2.waitKey(0)

    return mouth

def get_full_mask_occlusion(face, eyes):
    fx, fy, fw, fh = face
    _, ey, _, eh = eyes[0]

    mouth_x = fx + fw//20
    mouth_w = 18*fw//20
    mouth_y = ey + eh
    mouth_h = fy+fh-ey-eh

    return (mouth_x, mouth_y, mouth_w, mouth_h)

def get_small_occlusion(face, eyes):
    # covers a small area around the face
    fx, fy, fw, fh = face
    _, ey, _, eh = eyes[0]
    mouth_x = fx + 2*fw//9
    mouth_w = 5*fw//9
    mouth_y = fy + fh - 2*fh//8
    mouth_h = fh//4

    return (mouth_x, mouth_y, mouth_w, mouth_h)

def downsample(img):
    #downsample the image from 1280 x 960 to 80 x 60 using pyramidImage
    #NOTE: a single cv2.pyrDown() call only reduce the # of pixels to max(1/2 x original image size)
    #      thus, multiple calls is required to downsample it to desire size
    ds_img = cv2.pyrDown(img, dstsize=(int(img.shape[1]/2), int(img.shape[0]/2)))
    ds_img = cv2.pyrDown(ds_img, dstsize=(int(ds_img.shape[1]/2), int(ds_img.shape[0]/2)))
    ds_img = cv2.pyrDown(ds_img, dstsize=(int(ds_img.shape[1]/2), int(ds_img.shape[0]/2)))
    ds_img = cv2.pyrDown(ds_img, dstsize=(int(ds_img.shape[1]/2), int(ds_img.shape[0]/2)))
    return ds_img

def occlude_sample(occlusion_type, filename):
    img = cv2.cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)   
    crop_img = img[378:1338, 581:1861]
    estimate_mouth(crop_img, occlusion_type, None, True)

# expression code to directory name mapping
expr_dict = dict(
    N='neutral',
    A='angry',
    F='fear',
    HC='happy_closed',
    HO='happy_open',
)

def occlude_all(occlusion_type):
    files_dir = os.getcwd()
    source = cfg['source_folder']
    dest = os.path.join(cfg['dest_folder_occlusion'], occlusion_type)
    img_extensions = cfg['valid_img_extensions']

    try: os.mkdir(dest)
    except: pass

    num_failed = 0

    os.chdir(source)
    cwd = os.listdir(source)
    # iterate over directories in source dir
    for path in cwd:
        # print(full_path)
        # print(os.getcwd())
        # exit()
        if not os.path.isdir(path): 
            continue
        full_path = os.path.join(source, path)

        os.chdir(dest)
        try: os.mkdir(path)
        except: pass
        
        # iterate over imgs in this subject's dir
        os.chdir(full_path)
        for path2 in os.listdir(os.getcwd()):
            os.chdir(full_path)
            if not os.path.isfile(path2): continue
            
            # confirm that it's an img file
            i_dot = path2.rfind('.')
            if i_dot == -1: continue
            ext = path2[i_dot+1:]
            if ext not in img_extensions: continue

            # occlude img
            img = cv2.cvtColor(cv2.imread(path2), cv2.COLOR_BGR2GRAY)

            img = img[378:1338, 581:1861]
            occlusion = estimate_mouth(img, occlusion_type, files_dir)
            if occlusion is None:
                num_failed += 1
                continue

            x, y, w, h = occlusion
            img[y:y+h, x:x+w] = 0
            img = downsample(img)

            # save to dest folder
            os.chdir(dest)
            os.chdir(path)
            cv2.imwrite(f'{path2[:i_dot]}.jpg', img)

            # # plotting for testing purposes
            # plt.imshow(img, cmap='gray')

            # plt.show()

        os.chdir(source)
    
    print(f'Total number of imgs failed = {num_failed}')


# definition of types of occlusion, ordered by amt of coverage
FM = 'full_mask'
MO = 'mouth_only'


if __name__ == "__main__":
    # update config.py with source (db) and destination folders to use occlude_all
    # occlude_all(FM)

    # update filename with an img of a face from the db to see the occlusion sample
    # you can select either FM (full_mask) or MO (mouth_only) for occlusion type
    filename = 'people2.jpg'
    occlude_sample(MO, filename)