import os
import cv2
import random
import xml.etree.ElementTree as ET  # Libreria per leggere i file XML

# --- CONFIGURAZIONI INIZIALI ---
IMG_DIR = "archive/train/JPEGImages"       
LABEL_DIR = "archive/train/Annotations"     

OUT_DIR_UMANI = "dataset_ritagliato/train/human/"
OUT_DIR_SFONDO = "dataset_ritagliato/train/notHuman/"

TARGET_W = 64
TARGET_H = 128

os.makedirs(OUT_DIR_UMANI, exist_ok=True)
os.makedirs(OUT_DIR_SFONDO, exist_ok=True)

def si_sovrappongono(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    intersect_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    intersect_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    
    return intersect_x > 0 and intersect_y > 0

count_umani = 0
count_sfondo = 0

print("Inizio l'estrazione dei ritagli da formato XML (Pascal VOC)...")

for filename in os.listdir(IMG_DIR):
    if not filename.endswith(('.jpg', '.png')):
        continue
        
    img_path = os.path.join(IMG_DIR, filename)
    # Ora cerchiamo il file .xml con lo stesso nome dell'immagine
    xml_path = os.path.join(LABEL_DIR, filename.rsplit('.', 1)[0] + '.xml')
    
    if not os.path.exists(xml_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue
        
    img_h, img_w = img.shape[:2]
    box_persone = []

    # --- FASE 1: ESTRAZIONE DEGLI UMANI (LETTURA XML) ---
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Cerchiamo tutti i tag <object> dentro l'XML
    for obj in root.findall('object'):
        name = obj.find('name').text.lower()
        
        # ATTENZIONE: Controlla che nel tuo XML la classe si chiami 'person'. 
        # A volte potrebbe essere 'human' o 'pedestrian'. In tal caso modificalo qui!
        if name != 'person': 
            continue
            
        bndbox = obj.find('bndbox')
        # L'XML ci dà già le coordinate esatte, dobbiamo solo convertirle in interi!
        x_min = int(float(bndbox.find('xmin').text))
        y_min = int(float(bndbox.find('ymin').text))
        x_max = int(float(bndbox.find('xmax').text))
        y_max = int(float(bndbox.find('ymax').text))
        
        # Correzione di sicurezza per non uscire dai bordi dell'immagine
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(img_w, x_max), min(img_h, y_max)
        
        box_persone.append((x_min, y_min, x_max, y_max))
        
        ritaglio = img[y_min:y_max, x_min:x_max]
        
        if ritaglio.size == 0:
            continue
            
        ritaglio_resized = cv2.resize(ritaglio, (TARGET_W, TARGET_H))
        
        salvataggio_path = os.path.join(OUT_DIR_UMANI, f"human_{count_umani}.jpg")
        cv2.imwrite(salvataggio_path, ritaglio_resized)
        count_umani += 1

    # --- FASE 2: ESTRAZIONE DELLO SFONDO ---
    # (Questa parte rimane identica, la matematica delle Bounding Box combacia perfettamente)
    tentativi_max = 20
    sfondi_estratti = 0
    
    for _ in range(tentativi_max):
        if sfondi_estratti >= 2:
            break
            
        rand_w = random.randint(50, 200)
        rand_h = int(rand_w * 2) 
        
        if rand_w >= img_w or rand_h >= img_h:
            continue
            
        rand_x_min = random.randint(0, img_w - rand_w)
        rand_y_min = random.randint(0, img_h - rand_h)
        rand_x_max = rand_x_min + rand_w
        rand_y_max = rand_y_min + rand_h
        
        box_random = (rand_x_min, rand_y_min, rand_x_max, rand_y_max)
        
        sovrapposto = False
        for box_p in box_persone:
            if si_sovrappongono(box_random, box_p):
                sovrapposto = True
                break
                
        if not sovrapposto:
            ritaglio_sfondo = img[rand_y_min:rand_y_max, rand_x_min:rand_x_max]
            
            if ritaglio_sfondo.size > 0:
                ritaglio_sfondo_resized = cv2.resize(ritaglio_sfondo, (TARGET_W, TARGET_H))
                salvataggio_sfondo_path = os.path.join(OUT_DIR_SFONDO, f"notHuman_{count_sfondo}.jpg")
                cv2.imwrite(salvataggio_sfondo_path, ritaglio_sfondo_resized)
                count_sfondo += 1
                sfondi_estratti += 1

print(f"\nOperazione completata!")
print(f"Persone estratte: {count_umani}")
print(f"Sfondi estratti: {count_sfondo}")