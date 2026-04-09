import os
import cv2

IMG_DIR = "cars"           

OUT_DIR_SFONDO = "cars_dat"

TARGET_W = 64
TARGET_H = 128
count_umani = 1000
os.makedirs(OUT_DIR_SFONDO, exist_ok=True)

for filename in os.listdir(IMG_DIR):
    if not filename.endswith(('.jpg', '.png')):
        continue
        
    img_path = os.path.join(IMG_DIR, filename)

    img = cv2.imread(img_path)
    if img is None:
        continue
        
    img_h, img_w = img.shape[:2]

    ritaglio_resized = cv2.resize(img, (TARGET_W, TARGET_H))
    salvataggio_path = os.path.join(OUT_DIR_SFONDO, f"car_{count_umani}.jpg")
    cv2.imwrite(salvataggio_path, ritaglio_resized)
    count_umani = count_umani + 1