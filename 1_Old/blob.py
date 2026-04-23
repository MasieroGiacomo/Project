import cv2
import numpy as np

def display_image_and_mask_cv(img, mask):
    # 1. Convertiamo subito la maschera (True/False) in 0/255 (Grigio)
    mask_visual = (mask.astype(np.uint8) * 255)  
    
    # 2. IL TRUCCO: Creiamo un "pennello" grosso e applichiamo la Chiusura (CLOSE)
    # Un kernel 15x15 o 21x21 è abbastanza grande per tappare i buchi nei corpi
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask_visual = cv2.morphologyEx(mask_visual, cv2.MORPH_CLOSE, kernel)
    
    # (Opzionale ma consigliato) Applichiamo anche una leggera apertura per togliere 
    # i pixel sparsi fuori dalle persone (rumore)
    kernel_piccolo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_visual = cv2.morphologyEx(mask_visual, cv2.MORPH_OPEN, kernel_piccolo)

    # 3. Aggiorniamo la maschera logica True/False basandoci sulla maschera appena "stuccata"
    mask_stuccata = mask_visual > 0

    img_copy = img.copy()  
    # Sfondo bianco dove la maschera stuccata è falsa
    img_copy[np.logical_not(mask_stuccata)] = [255, 255, 255]   
    
    # Convertiamo a 3 canali per l'unione
    mask_bgr = cv2.cvtColor(mask_visual, cv2.COLOR_GRAY2BGR)    # b&w
    combined = np.hstack((mask_bgr, img_copy.astype(np.uint8)))
    #Scv2.imshow('TFD: Maschera e Immagine', combined)
    cv2.imshow('Mask Stuccata', img_copy)
    #cv2.imshow('Copy', mask_bgr)
    cv2.imshow('video', img)

'''def display_image_and_mask_cv(img, mask):
    img_copy = img.copy()  
    # Sfondo bianco dove la maschera è falsa
    img_copy[np.logical_not(mask)] = [255, 255, 255]   
    # Convertiamo la maschera (True/False o 0/1) in un'immagine a scala di grigi standard (0/255)
    mask_visual = (mask.astype(np.uint8) * 255)  
    # Per incollarle vicine, devono avere lo stesso numero di canali. 
    # Convertiamo la maschera da 1 canale (Grigio) a 3 canali (BGR fasullo)
    mask_bgr = cv2.cvtColor(mask_visual, cv2.COLOR_GRAY2BGR)  
    # Uniamo le immagini orizzontalmente (hstack)
    combined = np.hstack((mask_bgr, img_copy.astype(np.uint8)))
    # Mostriamo la finestra di OpenCV
    cv2.imshow('TFD: Maschera e Immagine', combined)
'''

def find_blobs_static_bg(frame, bg_float, threshold, min_area, k_cl, k_op):
    # 1. Calcolo distanza L2 usando la tua funzione
    frame_float = frame.astype(float)
    dist = l2_distance(frame_float, bg_float)
    
    # 2. Creazione Maschera Binaria (0 o 255)
    mask_thresh = (dist > threshold).astype(np.uint8) * 255
    
    # 3. Operazioni Morfologiche per pulire il rumore e unire le persone
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_cl, k_cl))
    mask_clean = cv2.morphologyEx(mask_thresh, cv2.MORPH_CLOSE, kernel_close)
    
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_op, k_op))
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


def play_video(video_path):
    cap = cv2.VideoCapture(video_path)
    backSub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=50, detectShadows=True)
    frame_count=0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret or frame is None:
            # Release the Video if ret is false
            cap.release()
            print("Released Video Resource")
            break

        frame_count += 1
        if frame_count % 2 != 0:  # Display one frame every 2 (now 15 fps)
            continue
        cv2.imshow('frame', frame)
        #frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        fg_mask = backSub.apply(frame)
        cv2.imshow('mask', fg_mask)
        

        retval, mask_thresh = cv2.threshold(fg_mask, 240, 255, cv2.THRESH_BINARY)
        cv2.imshow('thresh', mask_thresh)
        
        # set the kernal
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # Apply erosion
        mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)
        cv2.imshow('ero', mask_eroded)

        contours, hierarchy = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print(contours)
        frame_ct = cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
        # Display the resulting frame
        cv2.imshow('Frame_final', frame_ct)
        
    
        #print(f"Height: {frame.shape[0]}")
        #print(f"Width: {frame.shape[1]}")
        # Stop playing when I click 'q' 
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


play_video('videos/attraversamento.mp4')

