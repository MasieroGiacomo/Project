import numpy as np
import cv2      
from matplotlib import pyplot as plt


# Initializations 
path = 'videos/parking.mp4'
num_frame_bg = 200
interpolation = 1   # 0 -> mean, 1 -> median
bg = []
idx = 0
cap = cv2.VideoCapture(path)

while(cap.isOpened() and idx < num_frame_bg):
    ret, frame = cap.read()
    if ret and not frame is None:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame_gray.astype(float)
        # Getting all first n images
        bg.append(frame)
        idx += 1
    else:
        break
    
# Finding the background as the mean or median of first n frames
bg_interpolated = np.stack(bg, axis=0)
if interpolation == 0:
    bg_interpolated = np.mean(bg_interpolated, axis=0)
elif interpolation == 1:
    bg_interpolated = np.median(bg_interpolated, axis=0)

plt.title("Background")
plt.imshow(bg_interpolated, cmap='gray', vmin=0, vmax=255)
plt.show()

# Convert the array to an image 
bg_image_to_save = bg_interpolated.astype(np.uint8)

# Save the image
cv2.imwrite("Results/background.png", bg_image_to_save)
print("Background was saved as 'back.png'!")

cap.release()
cv2.destroyAllWindows()
print("Released Video Resource")