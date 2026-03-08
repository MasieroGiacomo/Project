import cv2
import numpy as np

def estrai_candidati(frame, backSub, min_area=500):
    """
    Applica la sottrazione dello sfondo e restituisce le bounding box dei blob validi.
    min_area: scarta i blob più piccoli di questi pixel (es. rumore o foglie).
    """
    # 1. Applica MOG2 (Il parametro learningRate=-1 fa calcolare l'alpha in automatico)
    fg_mask = backSub.apply(frame, learningRate=-1)
    
    # 2. Elimina le ombre (i pixel grigi di MOG2 diventano neri)
    _, mask_thresh = cv2.threshold(fg_mask, 240, 255, cv2.THRESH_BINARY)
    
    # 3. Pulizia morfologica (Erosione + Dilatazione = Apertura)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_clean = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)
    
    # 4. Trova i contorni SULLA MASCHERA PULITA
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bounding_boxes = []
    
    # 5. Filtra i contorni e calcola i rettangoli
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area: # Se la macchia è grande abbastanza per essere una persona...
            x, y, w, h = cv2.boundingRect(cnt)
            bounding_boxes.append((x, y, w, h))
            
    return bounding_boxes

def main():
    video_path = 'store.mp4'
    cap = cv2.VideoCapture(video_path)
    
    # Inizializziamo MOG2 impostando la history e la threshold (Sensibilità T)
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
    
    frame_count = 0
    
    # Dimensioni richieste dalla tua CNN (es. 64 larghezza, 128 altezza)
    CNN_INPUT_SIZE = (64, 128) 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Salta i frame per andare più veloci (es. elabora 1 frame ogni 3)
        frame_count += 1
        if frame_count % 2 != 0:
            continue
            
        # --- 1. TROVA I CANDIDATI ---
        boxes = estrai_candidati(frame, backSub, min_area=800)
        
        # --- 2. RITAGLIO E CLASSIFICAZIONE ---
        for (x, y, w, h) in boxes:
            
            # --- SAFE CROP: Previene crash se il box esce dallo schermo ---
            y_start = max(0, y)
            y_end = min(frame.shape[0], y + h)
            x_start = max(0, x)
            x_end = min(frame.shape[1], x + w)
            
            # Effettua il ritaglio matematico
            ritaglio_persona = frame[y_start:y_end, x_start:x_end]
            
            # Se il ritaglio è vuoto (es. errore ai bordi), saltalo
            if ritaglio_persona.size == 0:
                continue
                
            # Ridimensiona l'immagine per la CNN (ignora le proporzioni per ora)
            ritaglio_ridimensionato = cv2.resize(ritaglio_persona, CNN_INPUT_SIZE)
            
            # ---> QUI ANDRA' LA CHIAMATA ALLA TUA CNN <---
            # is_persona = mia_cnn.predict(ritaglio_ridimensionato)
            # if is_persona:
            #     ... controlla l'istogramma ...
            
            # Per ora, disegniamo un rettangolo BLU per vedere cosa abbiamo catturato
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, "Candidato", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Mostra il risultato finale
        cv2.imshow('Tracking Pipeline', frame)
        
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Eseguiamo il codice
if __name__ == "__main__":
    main()