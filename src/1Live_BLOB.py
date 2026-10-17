import os
os.environ["KERAS_BACKEND"] = "torch"
import numpy as np
import cv2
from keras.models import load_model

def main():

    ## PARAMETERS ##
    video_path = 'videos/wall.mp4'
    threshold = 30
    min_area = 170
    k_cl = 9
    k_op = 3
    k_dil = 3
    CNN_INPUT_SIZE = (64, 128)          # (width, height)
    model_type = 'Models/steve.keras'
    
    # ================================================ #
    ### 1. INITIALIZATIONS
    frame_count = 0
    cap = cv2.VideoCapture(video_path)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    disp_w = 350                                # larghezza desiderata della finestra
    disp_h = int(disp_w * vid_h / vid_w)        # altezza che mantiene le proporzioni
    
    for name in ['Mask', 'Frame with Detections']:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(name, disp_w, disp_h)

    ### 1b. BACKGROUND INITIALIZATION 
    background = cv2.imread("Results/background1.png", cv2.IMREAD_GRAYSCALE)
    if background is None:
        raise ValueError("Couldn't find the background")
    # cv2.imshow('Background', background)
    # Convert background to float for calculations
    bg_float = background.astype(float)
    
    ### 1c. LOAD OF THE CNN
    print("Loading CNN...")
    myModel = load_model(model_type)
    print("CNN ready!")
    
    # ================================================ #
    ### 2. VIDEO PROCESSING
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        # if frame_count % 2 != 0:
        #     continue

        # -------------------------------------------- #
        ## A. BLOB DETECTION    

        # Copy of the original frame for the CNN and visualization
        frame_colored = frame.copy()
        
        # Convert to grayscale and float for distance calculation
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   
        frame_float = frame_gray.astype(float) 

        # Compute the L2 distance between current frame and background   
        diff =  np.abs(frame_float - bg_float)
        # Threshold (obtain a boolean mask)
        mask = diff > threshold

        # Convert boolean mask to uint8 (0 or 255)
        mask_thresh = mask.astype(np.uint8) * 255
        
        # Morphological operations
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_cl, k_cl))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_op, k_op))
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (k_dil, k_dil))

        opened = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel_open)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=5)
        dilated = cv2.dilate(closed, kernel_dilate)

        # Detection of the contours 
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # -------------------------------------------- #
        ## B. CLASSIFICATION WITH CNN
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area: 
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Cut the detected blob from the original frame (colored)
                y_start, y_end = max(0, y), min(frame_colored.shape[0], y + h)
                x_start, x_end = max(0, x), min(frame_colored.shape[1], x + w)
                cut = frame_colored[y_start:y_end, x_start:x_end]
                
                if cut.size == 0:
                    continue
                    
                # Size-based filtering 
                ratio = h / float(w)
                if ratio < 1.2: 
                    # Skip the predicion and annotate as "Other"
                    cv2.rectangle(frame_colored, (x, y), (x+w, y+h), (0, 0, 255), 1)
                    cv2.putText(frame_colored, "Other (Horizontal)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    continue

                # Pre-processing for the CNN: resize and convert to RGB
                cut_rgb = cv2.cvtColor(cut, cv2.COLOR_BGR2RGB)
                cut_resized = cv2.resize(cut_rgb, CNN_INPUT_SIZE)
                # The model expects a batch dimension (1, height, width, channels)
                cut_batch = np.expand_dims(cut_resized, axis=0)     

                # Prediction with the CNN
                pred = myModel.predict(cut_batch, verbose=0) 
                
                # Annotations
                if pred[0][0] < 0.5:    # Human (green)
                    cv2.rectangle(frame_colored, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame_colored, f"Person {(1 - pred[0][0])*100:.0f}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:                   # Other (red)
                    cv2.rectangle(frame_colored, (x, y), (x+w, y+h), (0, 0, 255), 1)
                    cv2.putText(frame_colored, "Other", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # -------------------------------------------- #
        ## C. VISUALIZATION
        cv2.imshow('Mask', dilated)
        cv2.imshow('Frame with Detections', frame_colored) 
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # ================================================ #
    ### 3. CLOSING
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()