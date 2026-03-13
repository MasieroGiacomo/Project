import numpy as np
import cv2
from matplotlib import pyplot as plt
# Import additional library to properply play videos on jupyter notebook
from IPython.display import clear_output

vid = 2
if vid==0:
    path = 'rilevamento-intrusioni-video.avi'
elif vid==1:
    path = 'store.mp4'
elif vid == 2:
    path = 'amb.mp4'
elif vid == 3:
    path = 'city.mp4'

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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
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
    mask_bgr = cv2.cvtColor(mask_visual, cv2.COLOR_GRAY2BGR)  
    combined = np.hstack((mask_bgr, img_copy.astype(np.uint8)))
    cv2.imshow('TFD: Maschera e Immagine', combined)

# Defining a variable interpolation for mean or median functions
interpolation = np.median # or np.mean

# Loading Video
num_frame_bg = 100
bg = []
idx = 0
cap = cv2.VideoCapture(path)

# Initialize the background image

while(cap.isOpened() and idx < num_frame_bg):
    ret, frame = cap.read()
    if ret and not frame is None:
        frame = frame.astype(float)
        # Getting all first n images
        bg.append(frame)
        idx += 1
    else:
        break
    
# Finding the background as the mean or median of first n frames
bg_interpolated = np.stack(bg, axis=0)
# Median
bg_interpolated = interpolation(bg_interpolated, axis=0)
    
plt.title("Background")
plt.imshow(cv2.cvtColor(bg_interpolated.astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.show()

cap.release()
cv2.destroyAllWindows()
print("Released Video Resource")


# Loading Video
threshold = 50
cap2 = cv2.VideoCapture(path)
distance = max_distance

# Background subtraction
while(cap2.isOpened()):
    ret, frame = cap2.read()
    if ret and frame is not None:
        
        # Facciamo i calcoli usando i float
        frame_float = frame.astype(float)
        mask = distance(frame_float, bg_interpolated) > threshold
        
        # Passiamo il frame alla funzione
        display_image_and_mask_cv(frame, mask)
        
        # IL PEZZO MANCANTE FONDAMENTALE:
        # 30 ms di pausa per aggiornare la finestra (circa 30 fps)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
            
        # Pulisce l'output del notebook se stai stampando testo
        clear_output(wait=True)
    else:
        break

# ATTENZIONE QUI: cap2, non cap!
cap2.release()
cv2.destroyAllWindows()
print("Released Video Resource")