import numpy as np
import cv2
from keras.models import load_model
import func

# Defining a TFD based on a distance function and a threshold
def two_frame_difference(frame, previous_frame, distance=func.l1_distance, th=5):
    dist = distance(frame, previous_frame)
    mask = dist > th
    func.display_image_and_mask_cv(frame,mask)



# --- FUNZIONE DI RICERCA BLOB CON BACKGROUND STATICO (L2) ---
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
    idx = 0
    video_path = 'videos/attraversamento.mp4'
    cap = cv2.VideoCapture(video_path)
    
    background = cv2.imread("back.png", cv2.IMREAD_GRAYSCALE)
    if background is None:
        raise ValueError("ERRORE: Immagine 'back.png' non trovata!")
    bg_float = background.astype(float)
    
    frame_count = 0
    print("Caricamento Rete Neurale in corso...")
    modello_pedoni = load_model('modello_pedoni.keras')
    print("Rete pronta!")
    
    CNN_INPUT_SIZE = (64, 128) # Attenzione: per cv2.resize è (width, height)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 2 != 0:
            continue    
        # =========================================================
        # 1. PREPARAZIONE DEI FRAME
        # =========================================================
        
        # Copia intatta (uint8, BGR) per disegnare e per la Rete Neurale
        frame_display = frame.copy() 
        
        # Versione in Scala di Grigi e Float PER LA MATEMATICA dei blob
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_float = frame_gray.astype(float)
        
        if idx == 0:
            # Al primo giro, inizializza semplicemente il previous_frame
            previous_frame = frame_float.copy()
            idx += 1
            continue # Salta direttamente al prossimo frame
            
        frame_count += 1      
              
        # =========================================================
        # 2. RICERCA BLOB (Usa i frame float in scala di grigi)
        # =========================================================
        # Assicurati che find_blobs_static_bg non faccia un'ulteriore 
        # conversione astype(float), anche se non darebbe errore.
        boxes = find_blobs_static_bg(frame_float, previous_frame, threshold=30, min_area=150)
        
        # =========================================================
        # 3. RITAGLIO E CLASSIFICAZIONE
        # =========================================================
        for (x, y, w, h) in boxes:
            
            # SAFE CROP: Ritaglia dal frame_display (che è a colori e uint8!)
            y_start, y_end = max(0, y), min(frame_display.shape[0], y + h)
            x_start, x_end = max(0, x), min(frame_display.shape[1], x + w)
            cut = frame_display[y_start:y_end, x_start:x_end]
            
            if cut.size == 0:
                continue
                
            # PRE-PROCESSING PER LA CNN
            cut_rgb = cv2.cvtColor(cut, cv2.COLOR_BGR2RGB) # Ora funziona!
            cut_resized = cv2.resize(cut_rgb, CNN_INPUT_SIZE)
            cut_batch = np.expand_dims(cut_resized, axis=0)
            
            rapporto_forma = h / float(w)

            if rapporto_forma < 1.2: 
                # Disegna su frame_display!
                cv2.rectangle(frame_display, (x, y), (x+w, y+h), (0, 0, 255), 1)
                cv2.putText(frame_display, "Other (Horizontal)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                continue

            # CHIAMATA ALLA CNN
            predizione = modello_pedoni.predict(cut_batch, verbose=0) 
            
            if predizione[0][0] < 0.5:
                cv2.rectangle(frame_display, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame_display, f"Person {(1- predizione[0][0])*100:.0f}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame_display, (x, y), (x+w, y+h), (0, 0, 255), 1)
                cv2.putText(frame_display, "Other", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # =========================================================
        # 4. AGGIORNAMENTO E VISUALIZZAZIONE
        # =========================================================
        
        # Mostra a schermo l'immagine a colori con i rettangoli disegnati
        cv2.imshow('Tracking Pipeline', frame_display)
        
        # Aggiorna previous_frame con il frame_float ATTUALE per il prossimo giro
        # Usa .copy() per assicurarti che non ci siano reference sballate in memoria
        previous_frame = frame_float.copy() 
        
        idx += 1
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

