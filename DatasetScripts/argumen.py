import os
import cv2
import random

# --- CONFIGURAZIONI ---
IN_DIR = "dataset/train/Human/"
TARGET_FINALE = 3345 

files = [f for f in os.listdir(IN_DIR) if f.endswith(('.jpg', '.png'))]
num_originali = len(files)
mancanti = TARGET_FINALE - num_originali

if mancanti <= 0:
    print("Hai già raggiunto il target di immagini!")
else:
    print(f"Immagini originali: {num_originali}. Ne genererò altre {mancanti}.")

    count_generati = 0
    indice = 0

    while count_generati < mancanti:
        # Peschiamo un'immagine dalla lista in modo ciclico
        filename = files[indice % num_originali]
        img_path = os.path.join(IN_DIR, filename)
        img = cv2.imread(img_path)

        if img is not None:
            # Scegliamo l'operazione da fare alternandole (0, 1, 2)
            operazione = count_generati % 2

            if operazione == 0:
                # 1. Flip Orizzontale (Destra-Sinistra)
                img_aug = cv2.flip(img, 1)
                prefisso = "flipH_"
                
            elif operazione == 1:
                # 2. Modifica Contrasto
                # Scegliamo un valore alpha casuale per avere contrasti sempre diversi
                # (es. tra 0.6 per basso contrasto e 1.6 per alto contrasto)
                alpha = random.uniform(0.6, 1.6) 
                # convertScaleAbs applica la formula: pixel_nuovo = pixel_vecchio * alpha
                img_aug = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
                prefisso = f"contrast_"

            # Salviamo la nuova immagine
            nuovo_nome = f"{prefisso}{count_generati}_{filename}"
            cv2.imwrite(os.path.join(IN_DIR, nuovo_nome), img_aug)
            
            count_generati += 1

        indice += 1

    # Ricalcoliamo il totale alla fine
    totale_finale = len([f for f in os.listdir(IN_DIR) if f.endswith(('.jpg', '.png'))])
    print(f"\nOperazione completata! Ora hai {totale_finale} immagini in totale nella cartella.")