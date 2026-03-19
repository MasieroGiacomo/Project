import os
import cv2
import random
import xml.etree.ElementTree as ET

# --- CONFIGURAZIONI INIZIALI ---
IMG_DIR = "archive/Train/JPEGImages"       
LABEL_DIR = "archive/Train/Annotations"     

OUT_DIR_UMANI = "dataset_cut/train/human/"
OUT_DIR_SFONDO = "dataset_cut/train/notHuman/"

TARGET_W = 64
TARGET_H = 128
MARGIN_SICUREZZA = 20 # Pixel di "cuscinetto" attorno alle persone per evitare di tagliare braccia/gambe

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

print("Inizio l'estrazione dei ritagli...")

for filename in os.listdir(IMG_DIR):
    if not filename.endswith(('.jpg', '.png')):
        continue
        
    img_path = os.path.join(IMG_DIR, filename)
    xml_path = os.path.join(LABEL_DIR, filename.rsplit('.', 1)[0] + '.xml')
    
    if not os.path.exists(xml_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue
        
    img_h, img_w = img.shape[:2]
    
    # Useremo questa lista per memorizzare le zone da EVITARE per lo sfondo (con margine)
    zone_off_limits = []
    umani_estratti_qui = 0

    # --- FASE 1: ESTRAZIONE DEGLI UMANI ---
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    for obj in root.findall('object'):
        name = obj.find('name').text.lower()
        bndbox = obj.find('bndbox')
        
        x_min = int(float(bndbox.find('xmin').text))
        y_min = int(float(bndbox.find('ymin').text))
        x_max = int(float(bndbox.find('xmax').text))
        y_max = int(float(bndbox.find('ymax').text))
        
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(img_w, x_max), min(img_h, y_max)

        # Creiamo un'area espansa per evitare che i crop di sfondo catturino pezzi di questa persona
        safe_x_min = max(0, x_min - MARGIN_SICUREZZA)
        safe_y_min = max(0, y_min - MARGIN_SICUREZZA)
        safe_x_max = min(img_w, x_max + MARGIN_SICUREZZA)
        safe_y_max = min(img_h, y_max + MARGIN_SICUREZZA)
        
        # Registriamo l'area off-limits (indipendentemente dalla classe, così evitiamo anche auto/cani di sfondo)
        zone_off_limits.append((safe_x_min, safe_y_min, safe_x_max, safe_y_max))
        
        if name == 'person': 
            ritaglio = img[y_min:y_max, x_min:x_max] # Il ritaglio dell'umano resta preciso, senza margini
            if ritaglio.size == 0:
                continue
                
            ritaglio_resized = cv2.resize(ritaglio, (TARGET_W, TARGET_H))
            salvataggio_path = os.path.join(OUT_DIR_UMANI, f"human_{count_umani}.jpg")
            cv2.imwrite(salvataggio_path, ritaglio_resized)
            count_umani += 1
            umani_estratti_qui += 1

    # --- FASE 2: ESTRAZIONE DELLO SFONDO ---
    # Cerchiamo di estrarre tanti sfondi quanti sono gli umani nell'immagine.
    # Se non ci sono umani, proviamo a estrarre 1 sfondo per sfruttare comunque l'immagine.
    target_sfondi = max(1, umani_estratti_qui)
    
    tentativi_max = 50 # Aumentato per avere più chance in immagini affollate
    sfondi_estratti = 0
    
    for _ in range(tentativi_max):
        if sfondi_estratti >= target_sfondi:
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
        # Controlliamo la sovrapposizione con le zone OFF-LIMITS (che includono il margine di sicurezza)
        for box_off in zone_off_limits:
            if si_sovrappongono(box_random, box_off):
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