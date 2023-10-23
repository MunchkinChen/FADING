from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import os
from torchvision import transforms
import json

def get_instance_prompt(dreambooth_dir):
    json_path = os.path.join(dreambooth_dir, "model_config.json")
    with open(json_path, 'r') as file:
        model_config = json.load(file)
    return model_config['instance_prompt']

# get_instance_prompt('saved_model/de-id/FFHQ_512_00006_100')
#%%
def mydisplay(img):
    plt.axis('off')
    plt.imshow(img)
    plt.show()

def load_image(p, arr=False, resize=None):
    '''
    Function to load images from a defined path
    '''
    ret = Image.open(p).convert('RGB')
    if resize is not None:
        ret = ret.resize((resize[0],resize[1]))
    if not arr:
        return ret
    return np.array(ret)



def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def tensor_to_img(tensor,arr=False):
    tmp = tensor.clone().squeeze(0).cpu()
    tfpil = transforms.ToPILImage()
    tmp = tfpil(tmp)
    # tmp = (tmp+1)*0.5
    if arr:
        tmp = np.array(tmp)
    return tmp


#%%
def image_grid(imgs_, rows=None, cols=None, sort_file_filter=None, remove_filter=None, border=0, resize=None):
    if isinstance(imgs_, str) or (isinstance(imgs_, list) and isinstance(imgs_[0], str)):
        if isinstance(imgs_, str):
            # imgs 是一个dir
            files = os.listdir(imgs_)

            if remove_filter:
                files = remove_filter(files)

            if sort_file_filter:
                files = sorted(files, key=sort_file_filter)

            files = [os.path.join(imgs_, f) for f in files]
        else:
            # imgs 是一个dir的list
            files = imgs_

        print(files)

        imgs = []
        for f in files[:]:
            img = load_image(f,resize=resize)
            imgs.append(img)

    elif isinstance(imgs_, np.ndarray):
        # imgs 是一个ndarray
        imgs = [Image.fromarray(i) for i in imgs_]

    else:
        # imgs 是一个PIL的list
        imgs = imgs_[:]

    if not rows or not cols:
        rows = 1
        cols = len(imgs)

    assert len(imgs) == rows * cols

    w, h = imgs[-1].size
    grid = Image.new('RGB', size=(cols * w + (cols-1)*border, rows * h + (rows-1)*border), color='white')
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * (w+border), i // cols * (h+border)))
    return grid

#%%
def sort_by_num(separator='-'):
    def sort_by_num_(x):
        return int(x.split(separator, 1)[0])
    return sort_by_num_
def remove_filter(files):
    ret_files = []
    for f in files:
        if f[0]!= '.' and f.split('.')[0] not in ['1','8','17']:
            ret_files.append(f)
    return ret_files
def tmp(x):
    num = int(x[:5])
    return num


def get_person_placeholder(age=None, predicted_gender=None):
    if predicted_gender is not None:
        if age and age <= 15:
            person_placeholder = ['boy', 'girl'][predicted_gender == 'Female' or predicted_gender == 1]
        else: # init age > 15 或者根本没有init age
            person_placeholder = ['man','woman'][predicted_gender == 'Female' or predicted_gender == 1]
    else:
        if age and age <= 15:
            person_placeholder = "child"
        else:
            person_placeholder = "person"
    return person_placeholder