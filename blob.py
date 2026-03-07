import cv2


def play_video(video_path):
    cap = cv2.VideoCapture(video_path)
    backSub = cv2.createBackgroundSubtractorMOG2()
    frame_count=0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret or frame is None:
            # Release the Video if ret is false
            cap.release()
            print("Released Video Resource")
            break
        frame_count += 1
        if frame_count % 2 != 0:  # Display every second frame
            continue

        cv2.imshow('frame', frame)

        fg_mask = backSub.apply(frame)
        cv2.imshow('mask', fg_mask)
        contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print(contours)
        frame_ct = cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
        # Display the resulting frame
        cv2.imshow('Frame_final', frame_ct)

        retval, mask_thresh = cv2.threshold( fg_mask, 240, 255, cv2.THRESH_BINARY)
        cv2.imshow('thresh', mask_thresh)

        # set the kernal
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # Apply erosion
        mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)
        cv2.imshow('ero', mask_eroded)

        #print(f"Height: {frame.shape[0]}")
        #print(f"Width: {frame.shape[1]}")
        # Stop playing when entered 'q' from keyboard
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


play_video('store.mp4')

