import cv2
import numpy as np
from keras.models import load_model

def searchCandidate(frame, backSub, min_area=200): # <-- min_area abbassata!
    # 1. Applica MOG2 (learningRate=-1 = adattamento automatico)
    fg_mask = backSub.apply(frame, learningRate=-1)
    
    # 2. Elimina le ombre (MOG2 colora le ombre di grigio 127, noi teniamo solo il bianco 255)
    _, mask_thresh = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)
    
    # 3. Pulizia Morfologica "Soft" per non cancellare pedoni lontani
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) # kernel più piccolo
    mask_clean = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel_open)
    
    # NOVITA': Una Chiusura per unire i pixel dello stesso pedone (braccia, gambe)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_close)
    
    # (Opzionale) Se vuoi fare debug della maschera finale, scommenta la riga sotto
    cv2.imshow('MOG2 Maschera Pulita', mask_clean)
    
    # 4. Trova i contorni
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bounding_boxes = []
    
    # 5. Filtra per area
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area: 
            x, y, w, h = cv2.boundingRect(cnt)
            bounding_boxes.append((x, y, w, h))
            
    return bounding_boxes

def main():
    video_path = 'videos/uni5.mp4'
    cap = cv2.VideoCapture(video_path)
    
    # MOG2 Tuning per pedoni lontani: varThreshold abbassata a 16
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    
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
            
        # 1. Troviamo i box in movimento (MOG2)
        boxes = searchCandidate(frame, backSub, min_area=200)
        
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

            # CHIAMATA ALLA CNN
            predizione = modello_pedoni.predict(cut_batch, verbose=0) # verbose=0 nasconde i log di logica in console
            
            if predizione[0][0] > 0.5:
                # E' UN UMANO (Verde)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Persona {predizione[0][0]*100:.0f}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # E' SFONDO IN MOVIMENTO (Rosso - es. foglie, carrelli)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
                cv2.putText(frame, "Rumore", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        cv2.imshow('Tracking Pipeline', frame)
        
        # Aumentiamo un po' l'attesa per vedere il video a velocità umana
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()