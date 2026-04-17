import numpy as np
import cv2
import func

def main():
    video_path = 'videos/attraversamento.mp4'
    threshold = 30
    min_area = 170
    k_cl = 9
    k_op = 3
    k_dil = 3

    cap = cv2.VideoCapture(video_path)
    
    background = cv2.imread("back.png", cv2.IMREAD_GRAYSCALE)
    if background is None:
        raise ValueError("ERRORE: Immagine 'back.png' non trovata!")
    bg_float = background.astype(float)
    
    frame_count = 0
    
    CNN_INPUT_SIZE = (64, 128) 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Salvo una copia a colori per poterci disegnare sopra i rettangoli colorati
        frame_display = frame.copy()
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   
        frame_float = frame.astype(float) 
        frame_count += 1
            
        # 1. Calcolo distanza L2 usando la tua funzione
        #frame_float = frame.astype(float)
        #dist = func.l2_distance(frame_float, bg_float)
        diff = func.l2_distance(frame_float, bg_float) 
        mask = diff > threshold

        # 2. Creazione Maschera Binaria (0 o 255)
        #mask_thresh = (dist > threshold).astype(np.uint8) * 255 
        mask_thresh = mask.astype(np.uint8) * 255
        
        # 3. Operazioni Morfologiche per pulire il rumore e unire le persone
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_cl, k_cl))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_op, k_op))
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (k_dil, k_dil))

        #mask_clean = cv2.morphologyEx(mask_thresh, cv2.MORPH_CLOSE, kernel_close, iterations=3)
        #mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel_open)

        # Closing
        opened = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel_open)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=5)
        # Opening
        
        # Dilation
        dilated = cv2.dilate(closed, kernel_dilate)

        # 4. Ricerca dei Contorni
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bounding_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area: 
                x, y, w, h = cv2.boundingRect(cnt)
                bounding_boxes.append((x, y, w, h+5))  
                #print(f"Rilevato blob con area {area} in posizione ({x}, {y}, {w}, {h})")
                # Disegno il rettangolo verde sul frame originale a colori
                cv2.rectangle(frame_display, (x, y), (x+w, y+h+5), (0, 255, 0), 2)

        # 5. MOSTRA I RISULTATI A SCHERMO
        cv2.imshow('Maschera Pulita', dilated)
        cv2.imshow('Rilevamenti', frame_display) # Mostra anche l'originale con i box
        
        # --- FIX CRITICO ---
        # Aspetta 30 millisecondi. Se l'utente preme 'q', esce dal ciclo.
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

