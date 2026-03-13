import numpy as np
import cv2
from matplotlib import pyplot as plt

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


def display_image_and_mask_cv(img, mask):
    img_copy = img.copy()  
    # Sfondo bianco dove la maschera è falsa
    img_copy[np.logical_not(mask)] = [255, 255, 255]   
    # Convertiamo la maschera (True/False o 0/1) in un'immagine a scala di grigi standard (0/255)
    mask_visual = (mask.astype(np.uint8) * 255)  
    # Per incollarle vicine, devono avere lo stesso numero di canali. 
    # Convertiamo la maschera da 1 canale (Grigio) a 3 canali (BGR fasullo)
    mask_bgr = cv2.cvtColor(mask_visual, cv2.COLOR_GRAY2BGR)  
    # Uniamo le immagini orizzontalmente (hstack)
    combined = np.hstack((mask_bgr, img_copy.astype(np.uint8)))
    # Mostriamo la finestra di OpenCV
    cv2.imshow('TFD: Maschera e Immagine', combined)

# Doesn't work 
def display_image_and_mask(img, mask):
    img_copy = img.copy()
    img_copy[np.logical_not(mask)] = np.asarray([255,255,255])
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.axis('off')
    plt.imshow(mask.astype(np.uint8)*255, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1,2,2)
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img_copy.astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.show()

# Defining a TFD based on a distance function and a threshold
def two_frame_difference(frame, previous_frame, distance=l1_distance, th=5):
    dist = distance(frame, previous_frame)
    mask = dist > th
    display_image_and_mask_cv(frame,mask)

# Assigning to variable distance a distance function. Change this variable to test other distance functions
distance = mahalanobis_distance

# Defining a threshold
threshold = 60

# Loading video
idx = 0
cap = cv2.VideoCapture('store.mp4')

# Computing a TFD over frames
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret and not frame is None:
        if idx == 0:
            # If first frame initializing  previous frame
            previous_frame = frame.astype(float)
        else:
            frame = frame.astype(float)
            # Calculating a TFD
            two_frame_difference(frame, previous_frame, distance, threshold)
            # Updating previous frame
            previous_frame = frame
        idx += 1
    else:
        break
    if cv2.waitKey(1) == ord('q'):
            break
    
cap.release()
cv2.destroyAllWindows()
print("Released Video Resource")