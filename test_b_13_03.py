import numpy as np
import cv2
from matplotlib import pyplot as plt
# Import additional library to properply play videos on jupyter notebook
from IPython.display import clear_output
import func

vid = 1
if vid==0:
    videoP = 'videos/rilevamento-intrusioni-video.avi'
elif vid==1:
    videoP = 'videos/p3.mp4'
elif vid == 2:
    videoP = 'videos/amb.mp4'
elif vid == 3:
    videoP = 'videos/city.mp4'


# Loading Video
threshold = 55
cap2 = cv2.VideoCapture(videoP)
distance = func.l2_distance
frame_count = 0

# 1. Carica l'immagine salvata in precedenza
background = cv2.imread("back.png")
# Background subtraction
while(cap2.isOpened()):
    ret, frame = cap2.read()
    if ret and frame is not None:
    
    # Elaborate every 2 frames to speed up (15 fps)
        frame_count += 1
        if frame_count % 2 != 0:
            continue
        # Facciamo i calcoli usando i float
        frame_float = frame.astype(float)
        bg_interpolated = background.astype(float)

        mask = distance(frame_float, bg_interpolated) > threshold
        
        # Passiamo il frame alla funzione
        func.display_image_and_mask_cv(frame, mask)
        
        # IL PEZZO MANCANTE FONDAMENTALE:
        # 30 ms di pausa per aggiornare la finestra (circa 30 fps)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
            
        # Pulisce l'output del notebook se stai stampando testo
        clear_output(wait=True)
    else:
        break

cap2.release()
cv2.destroyAllWindows()
print("Released Video Resource")