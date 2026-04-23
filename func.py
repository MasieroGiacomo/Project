import numpy as np

def l1_distance(img1, img2):
    diff = np.abs(img1-img2)
    print(len(img1.shape))
    if img1.shape[-1] ==  3 and len(img1.shape)==3:
        diff = np.sum(diff, axis=-1)
    return diff

def l2_distance(img1, img2):
    sq_dist = (img1-img2)**2
    if img1.shape[-1] ==  3 and len(img1.shape)==3:
        sq_dist = np.sum(sq_dist,axis=-1)
    diff = np.sqrt(sq_dist)
    return diff

def max_distance(img1, img2):
    diff = np.abs(img1-img2)
    if img1.shape[-1] ==  3 and len(img1.shape)==3:
        diff = np.max(diff, axis=-1)
    return diff