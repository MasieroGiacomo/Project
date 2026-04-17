import numpy as np
import cv2
from keras.models import load_model
import func

def find_blobs_static_bg(frame, bg_float, threshold, min_area):
    # 1. Calcolo distanza L2 usando la tua funzione
    frame_float = frame.astype(float)
    dist = func.l2_distance(frame_float, bg_float)
    
    # 2. Creazione Maschera Binaria (0 o 255)
    mask_thresh = (dist > threshold).astype(np.uint8) * 255
    
    # 3. Operazioni Morfologiche per pulire il rumore e unire le persone
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask_clean = cv2.morphologyEx(mask_thresh, cv2.MORPH_CLOSE, kernel_close)
    
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel_open)

    cv2.imshow('Maschera Pulita', mask_clean)

    # 4. Ricerca dei Contorni
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bounding_boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area: 
            x, y, w, h = cv2.boundingRect(cnt)
            bounding_boxes.append((x, y, w, h+5))
            
    return bounding_boxes

def main():
    video_path = 'videos/parking.mp4'
    cap = cv2.VideoCapture(video_path)
    
    background = cv2.imread("Results/back.png", cv2.IMREAD_GRAYSCALE)
    if background is None:
        raise ValueError("ERRORE: Immagine 'back.png' non trovata!")
    bg_float = background.astype(float)
    
    frame_count = 0
    print("Caricamento Rete Neurale in corso...")
    modello_pedoni = load_model('Results/modello_pedoni.keras')
    print("Rete pronta!")
    
    CNN_INPUT_SIZE = (64, 128) # Attenzione: per cv2.resize è (width, height)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
        frame_count += 1
        if frame_count % 2 != 0:
            continue
            
        # 1. RICERCA BLOB
        boxes = find_blobs_static_bg(frame, bg_float, threshold=30, min_area=150)
        
        # 2. Ritaglio e Classificazione
        for (x, y, w, h) in boxes:
            
            # SAFE CROP
            y_start, y_end = max(0, y), min(frame.shape[0], y + h)
            x_start, x_end = max(0, x), min(frame.shape[1], x + w)
            cut = frame[y_start:y_end, x_start:x_end]
            
            if cut.size == 0:
                continue
                
            # PRE-PROCESSING PER LA CNN
            # A) Inverti i colori da BGR (OpenCV) a RGB (Keras)
            cut_rgb = cv2.cvtColor(cut, cv2.COLOR_BGR2RGB)
            # B) Ridimensiona
            cut_resized = cv2.resize(cut_rgb, CNN_INPUT_SIZE)
            # C) Crea il Batch (1, 128, 64, 3)
            cut_batch = np.expand_dims(cut_resized, axis=0)
            # Aggiungi questo controllo PRIMA di chiamare la CNN
            rapporto_forma = h / float(w)

            # Se il box è più largo che alto, o è troppo largo rispetto all'altezza (es. un'auto)
            if rapporto_forma < 1.2: 
                # Disegnalo direttamente come Rumore/Sfondo, salta la predizione
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
                cv2.putText(frame, "Other (Horizontal)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                continue

            # CHIAMATA ALLA CNN
            predizione = modello_pedoni.predict(cut_batch, verbose=0) # verbose=0 nasconde i log di logica in console
            
            if predizione[0][0] < 0.5:
                # E' UN UMANO (Verde)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {(1- predizione[0][0])*100:.0f}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # E' SFONDO IN MOVIMENTO (Rosso - es. foglie, carrelli)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
                cv2.putText(frame, "Other", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        cv2.imshow('Tracking Pipeline', frame)
        
        # Aumentiamo un po' l'attesa per vedere il video a velocità umana
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

