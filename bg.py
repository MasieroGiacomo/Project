import numpy as np
import cv2      
from matplotlib import pyplot as plt


## BACKGROUND INITIALIZATION
num_frame_bg = 200
bg = []
idx = 0
cap = cv2.VideoCapture('videos/p3Bis.mp4')
interpolation = np.median # or np.mean

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
# 1. Convertiamo l'array float in un'immagine standard 8-bit
bg_image_to_save = bg_interpolated.astype(np.uint8)

# 2. Salviamo l'immagine sul disco fisso (nella stessa cartella)
cv2.imwrite("back.png", bg_image_to_save)
print("Background salvato con successo in 'back.png'!")

cap.release()
cv2.destroyAllWindows()
print("Released Video Resource")