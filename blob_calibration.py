import numpy as np
import cv2
import func

def main():
    ### PARAMETERS 
    video_path = 'videos/attraversamento.mp4'
    threshold = 30
    min_area = 170
    k_cl = 9
    k_op = 3
    k_dil = 3

    # ================================================ #
    ### 1. INITIALIZATIONS
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    ### 1b. BACKGROUND INITIALIZATION
    background = cv2.imread("back.png", cv2.IMREAD_GRAYSCALE)
    if background is None:
        raise ValueError("ERRORE: Immagine 'back.png' non trovata!")
    bg_float = background.astype(float)

    # ================================================ #
    ### 2. VIDEO PROCESSING    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1

        # Copy of the original frame for visualization
        frame_display = frame.copy()
        # Convert to grayscale and float for distance calculation
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   
        frame_float = frame.astype(float) 
        
        # Compute the L2 distance between current frame and background 
        diff = func.l2_distance(frame_float, bg_float) 
        # Threshold (obtain a boolean mask)
        mask = diff > threshold

        # Convert boolean mask to uint8 (0 or 255)
        mask_thresh = mask.astype(np.uint8) * 255
        
        # Morphological operationsS
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_cl, k_cl))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_op, k_op))
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (k_dil, k_dil))

        # Opening
        opened = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel_open)
        # Closing
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=5)
        # Dilation
        dilated = cv2.dilate(closed, kernel_dilate)

        # Detection of the contoursS
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Drwaing bounding boxes around the detected blobs
        bounding_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area: 
                x, y, w, h = cv2.boundingRect(cnt)
                bounding_boxes.append((x, y, w, h))  # Added for the feet +5 at h 
                #print(f"Blob of area {area} at ({x}, {y}, {w}, {h})")
                cv2.rectangle(frame_display, (x, y), (x+w, y+h+5), (0, 255, 0), 2)

        # Visualization
        cv2.imshow('Maschera Pulita', dilated)
        cv2.imshow('Rilevamenti', frame_display) # Mostra anche l'originale con i box
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # ================================================ #
    ### 3. CLOSING
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

