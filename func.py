import numpy as np
import cv2


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

'''def display_image_and_mask_cv(img, mask):
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
'''

def display_image_and_mask_cv(img, mask):
    # 1. Convertiamo subito la maschera (True/False) in 0/255 (Grigio)
    mask_visual = (mask.astype(np.uint8) * 255)  
    
    # 2. IL TRUCCO: Creiamo un "pennello" grosso e applichiamo la Chiusura (CLOSE)
    # Un kernel 15x15 o 21x21 è abbastanza grande per tappare i buchi nei corpi
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask_visual = cv2.morphologyEx(mask_visual, cv2.MORPH_CLOSE, kernel)
    
    # (Opzionale ma consigliato) Applichiamo anche una leggera apertura per togliere 
    # i pixel sparsi fuori dalle persone (rumore)
    kernel_piccolo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_visual = cv2.morphologyEx(mask_visual, cv2.MORPH_OPEN, kernel_piccolo)

    # 3. Aggiorniamo la maschera logica True/False basandoci sulla maschera appena "stuccata"
    mask_stuccata = mask_visual > 0

    img_copy = img.copy()  
    # Sfondo bianco dove la maschera stuccata è falsa
    img_copy[np.logical_not(mask_stuccata)] = [255, 255, 255]   
    
    # Convertiamo a 3 canali per l'unione
    mask_bgr = cv2.cvtColor(mask_visual, cv2.COLOR_GRAY2BGR)    # b&w
    combined = np.hstack((mask_bgr, img_copy.astype(np.uint8)))
    #Scv2.imshow('TFD: Maschera e Immagine', combined)
    cv2.imshow('Mask Stuccata', img_copy)
    #cv2.imshow('Copy', mask_bgr)
    cv2.imshow('video', img)