import cv2
import numpy as np

def searchCandidate(frame, backSub, min_area=500):
    """
    Applica la sottrazione dello sfondo e restituisce le bounding box dei blob validi.
    min_area: scarta i blob più piccoli di questi pixel (es. rumore o foglie).
    """
    # 1. Applica MOG2 (learningRate=-1 means authomatic adaption rate)
    fg_mask = backSub.apply(frame, learningRate=-1)
    cv2.imshow('mask', fg_mask)
    
    # 2. Eliminate shadows 
    _, mask_thresh = cv2.threshold(fg_mask, 240, 255, cv2.THRESH_BINARY)
    cv2.imshow('thresh', mask_thresh)
    
    # 3. Binary morpholofycal operations to clean the mask (remove noise and small blobs)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_clean = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)
    cv2.imshow('ero', mask_clean)
    
    # 4. Find contours (blobs) 
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bounding_boxes = []
    
    # 5. Filter the blobs and create bb
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area: 
            x, y, w, h = cv2.boundingRect(cnt)
            bounding_boxes.append((x, y, w, h))
            
    return bounding_boxes

def main():
    video_path = 'store.mp4'
    cap = cv2.VideoCapture(video_path)
    
    # Initialization MOG2  (history and threshold)
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
    
    frame_count = 0
    
    # Required dimensions of the CNN
    CNN_INPUT_SIZE = (64, 128) 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Elaborate every 2 frames to speed up (15 fps)
        frame_count += 1
        if frame_count % 2 != 0:
            continue
            
        # --- 1. Find candidates ---
        boxes = searchCandidate(frame, backSub, min_area=800)
        
        # --- 2. Cut + Classification ---
        for (x, y, w, h) in boxes:
            
            # --- SAFE CROP: Previene crash se il box esce dallo schermo ---
            y_start = max(0, y)
            y_end = min(frame.shape[0], y + h)
            x_start = max(0, x)
            x_end = min(frame.shape[1], x + w)
            
            # 
            cut = frame[y_start:y_end, x_start:x_end]
            
            # Se il ritaglio è vuoto (es. errore ai bordi), saltalo
            if cut.size == 0:
                continue
                
            # Ridimensiona l'immagine per la CNN (ignora le proporzioni per ora)
            cut_resized = cv2.resize(cut, CNN_INPUT_SIZE)
            
            # ---> QUI ANDRA' LA CHIAMATA ALLA TUA CNN <---
            # is_persona = mia_cnn.predict(ritaglio_ridimensionato)
            # if is_persona:
            #     ... controlla l'istogramma ...
            
            # Draw a bb around the candidate
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, "Candidato", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Show
        cv2.imshow('Tracking Pipeline', frame)
        
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()