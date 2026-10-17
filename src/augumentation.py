import os
import cv2
import random


IN_DIR = "dataset/new"
fin_num = 2500 

files = [f for f in os.listdir(IN_DIR) if f.endswith(('.jpg', '.png'))]
in_num = len(files)
need = fin_num - in_num

if need <= 0:
    print("Already got all needed images")
else:
    count = 0
    index = 0

    while count < need:
        # Get an image
        filename = files[index % in_num]
        img_path = os.path.join(IN_DIR, filename)
        img = cv2.imread(img_path)

        if img is not None:
            # Choose the operation (horizontal flip / contrast change / Gaussian blur)
            op = count % 3

            if op == 0:     # Horizontal filp
                img_aug = cv2.flip(img, 1)
                prefix = "flipH_"
                
            elif op == 1:   # Contrast change
                alpha = random.uniform(0.6, 1.4) 
                img_aug = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
                prefix = f"contrast_"

            elif op == 2:   # Gaussian blur
                img_aug = cv2.GaussianBlur(img, (5,5), sigmaX = 0)
                prefix = f"gaussian_"

            # Saving
            new_name = f"{prefix}{count}_{filename}"
            cv2.imwrite(os.path.join(IN_DIR, new_name), img_aug)

            count += 1
        index += 1

    tot_num = len([f for f in os.listdir(IN_DIR) if f.endswith(('.jpg', '.png'))])
    print(f"Final number of images: {tot_num}")