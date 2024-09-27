import staintools
from pathlib import Path
import pandas as pd
import re
import os
import cv2 as cv
from shutil import copy
from matplotlib import pyplot as plt
import tqdm
import multiprocessing
from pathlib import Path

'''
Function description:
    Detect the blurred images and move them to a backup directory;
    Correct all the clear images into a unified color space;
'''
def image_convert(_img_path, _new_img_path):
    template_image = Path("HEAL/Pre_processing/n6.png")
    template_image = str(template_image.resolve())
    target = staintools.read_image(template_image)
    
    # Normalization process
    normalizer = staintools.StainNormalizer(method='macenko')
    normalizer.fit(target)

    # Process the image
    image = staintools.read_image(_img_path)
    image = staintools.LuminosityStandardizer.standardize(image)
    img = normalizer.transform(image)

    # Save the processed image
    success = cv.imwrite(_new_img_path, img)
    if success:
        print(f"Processed image saved to: {_new_img_path}")
    else:
        print(f"Failed to save image: {_new_img_path}")

def blur_color_processing(_root, _img_path, _img):
    print(f"Processing image: {_img_path}")
    img, fm = find_blur(_img_path)
    print(f"Blur score: {fm}")

    if fm <= 100:
        blur_path = create_new_folder(_root, "tiling", "tiling_blur")
        copy(_img_path, blur_path)
        print(f"Copied blurred image to: {blur_path}")
    else:
        _new_img_folder = create_new_folder(_root, "tiling", "tiling_macenko")
        _new_img_path = os.path.join(_new_img_folder, _img)
        image_convert(_img_path, _new_img_path)

    return (fm)

def pre_processing(extra_prefix=""):
    print("[INFO] Starting blur detection ...")
    cpu_num = multiprocessing.cpu_count()
    print(f"The CPU number of this machine is {cpu_num}")
    pool = multiprocessing.Pool(int(cpu_num))
    
    _image_path = f"HEAL_Workspace/tiling{extra_prefix}"
    for _root, _dir, _imgs in os.walk(_image_path):
        _imgs = [f for f in _imgs if not f[0] == '.']
        _dir[:] = [d for d in _dir if not d[0] == '.']
        
        for idx in range(len(_imgs)):
            _img = _imgs[idx]
            _img_path = os.path.join(_root, _img)
            pool.apply_async(blur_color_processing, (_root, _img_path, _img))

    pool.close()
    pool.join()
