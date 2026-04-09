# ================================
# IMPORT LIBRERIES
# ================================
import numpy as np                  
import cv2                          
from matplotlib import pyplot as plt 

# ================================
# OUTPUT FILE
# ================================
output_file = open('output_detection.txt', 'w')

output_file.write(
    f"{'Frame':<8}{'N_obj':<8}{'ID':<6}" 
    f"{'X':<6}{'Y':<6}{'W':<6}{'H':<6}" 
    f"{'Area':<10}{'Perim':<10}"
    f"{'AspectR':<10}{'Class'}\n"
)
output_file.write("-"*90 + "\n") 

# ====================================
# INITIAL BACKGROUND MODEL ESTIMATION
# ====================================

estimation = np.median   

def l1_distance(img1, img2):
    diff = np.abs(img1 - img2)
    return diff

num_frame_bg = 100        
bg = []
idx = 0                   

cap = cv2.VideoCapture('rilevamento-intrusioni-video.avi')

try: 
    while cap.isOpened() and idx < num_frame_bg:
        
        ret, frame = cap.read()
        if ret and frame is not None:
           
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
            frame = frame.astype(float) 
            
            bg.append(frame) 
            idx += 1 
        else:
            break

    cap.release()

    bg_estimated = estimation(np.stack(bg, axis=0), axis=0) 

    plt.title("Background")
    plt.imshow(bg_estimated.astype(np.uint8), cmap='gray')
    plt.show()

except KeyboardInterrupt:  
    cap.release()

# ================================
# VIDEO ANALYSIS
# ================================
threshold = 20                     
cap = cv2.VideoCapture('rilevamento-intrusioni-video.avi') 

alpha_min = 0.001
alpha_max = 0.05 
frame_count = 0 


try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame.astype(float)
        frame_count += 1

        # ================================
        # BACKGROUND SUBTRACTION
        # ================================
        diff = l1_distance(frame, bg_estimated) 
        mask = diff > threshold

        # ================================
        # SMALL BLOBS REMOVAL
        # ================================
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8) 

        mask_clean = np.zeros_like(mask, dtype=np.uint8) 
        min_area = 400 

        for i in range(1, num_labels):   
            if stats[i, cv2.CC_STAT_AREA] >= min_area: 
                mask_clean[labels == i] = 1 

        # ================================
        # MORPHOLOGICAL OPERATIONS
        # ================================
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

        # Closing
        closed = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=5)

        # Opening
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=3)

        # Dilation
        dilated = cv2.dilate(opened, kernel, iterations=2)


        # ================================
        # BACKGROUND ADAPTIVE UPDATE
        # ================================
        norm_diff = diff / (np.max(diff) + 1e-6)

        alpha = alpha_max * (1 - norm_diff) 
        alpha = np.clip(alpha, alpha_min, alpha_max) 
        
        background_mask = np.logical_not(dilated.astype(bool)) 
        
        bg_estimated[background_mask] = (
            (1 - alpha[background_mask]) * bg_estimated[background_mask]
            + alpha[background_mask] * frame[background_mask]
        )
         

        # ================================
        # CONTOURS DETECTION
        # ================================
        contours, hier = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        
        
        num_objects = len(contours) 

        # ================================
        # FRAMES PREPARATION FOR DRAWING
        # ================================
        frame_draw = frame.astype(np.uint8)
        frame_draw = cv2.cvtColor(frame_draw, cv2.COLOR_GRAY2BGR)

        object_id = 0 

        # ================================
        # OBJECTS ANALYSIS
        # ================================
        for contour in contours:

            area = cv2.contourArea(contour) 

            if area <= 400: 
                continue

            x, y, w, h = cv2.boundingRect(contour) 

            perimeter = cv2.arcLength(contour, True) 

            aspect_ratio = w / h if h > 0 else 0 
            
            classification = 'person' if area > 3000 else 'other' 
            
            # ================================
            # FALSE OBJECTS DETECTION
            # ================================
            if classification == 'other':

                blob_mask = np.zeros_like(frame, dtype=np.uint8) 
                cv2.drawContours(blob_mask, [contour], -1, 255, -1) 

                inside_pixels = frame[blob_mask.astype(bool)] 
        
                kernel_border = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                dilated_blob = cv2.dilate(blob_mask, kernel_border, iterations=1) 
                      

                border_mask = dilated_blob.astype(bool) & ~blob_mask.astype(bool)
                outside_pixels = frame[border_mask]

                if len(outside_pixels) > 0: 
                    mean_inside = np.mean(inside_pixels) 
                    mean_outside = np.mean(outside_pixels) 
                    mean_diff = np.abs(mean_inside - mean_outside) 

                    if mean_diff < 12: 
                        classification = 'false_other'
                    else:
                        classification = 'true_other'
                else:
                    classification = 'true_other'


            # ================================
            # COLOURS CLASSIFICATION
            # ================================
            if classification == 'person':
                color = (0, 255, 0)      # Green
            elif classification == 'true_other':
                color = (255, 255, 0)    # Cyan
            elif classification == 'false_other':
                color = (0, 0, 255)      # Red
            else:
                color = (255, 255, 0)

            cv2.drawContours(frame_draw, [contour], -1, color, 2) 
            
            output_file.write(
                f"{frame_count:<8}"
                f"{num_objects:<8}"
                f"{object_id:<6}"
                f"{x:<6}{y:<6}{w:<6}{h:<6}"
                f"{int(area):<10}"
                f"{perimeter:<10.1f}"
                f"{aspect_ratio:<10.2f}"
                f"{classification}\n"
            )
            

            object_id += 1 


        # ================================
        # VISUALIZATION
        # ================================
        cv2.imshow('mask', mask.astype(np.uint8) * 255) 
        cv2.imshow('frame', frame_draw)

        if cv2.waitKey(30) & 0xFF == 27: 
            break


    # ================================
    # FILE CLOSING
    # ================================
    cap.release() 
    cv2.destroyAllWindows() 
    output_file.close() 

except KeyboardInterrupt:
    cap.release()