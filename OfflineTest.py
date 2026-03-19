import cv2
import numpy as np
from keras.models import load_model
import func

# --- FUNZIONE DI RICERCA BLOB CON BACKGROUND STATICO (L2) ---
def find_blobs_static_bg(frame, bg_float, threshold=55, min_area=200):
    # 1. Calcolo distanza L2 usando la tua funzione
    frame_float = frame.astype(float)
    dist = func.l2_distance(frame_float, bg_float)
    
    # 2. Creazione Maschera Binaria (0 o 255)
    mask_thresh = (dist > threshold).astype(np.uint8) * 255
    
    # 3. Operazioni Morfologiche per pulire il rumore e unire le persone
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_clean = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel_open)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_close)
    
    # 4. Ricerca dei Contorni
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bounding_boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area: 
            x, y, w, h = cv2.boundingRect(cnt)
            bounding_boxes.append((x, y, w, h))
            
    return bounding_boxes

def main():
    # --- SCELTA VIDEO E PATH ---
    vid = 1
    if vid == 0: videoP = 'videos/rilevamento-intrusioni-video.avi'
    elif vid == 1: videoP = 'videos/p3Bis.mp4'
    elif vid == 2: videoP = 'videos/amb.mp4'
    elif vid == 3: videoP = 'videos/city.mp4'

    print(f"Inizializzazione elaborazione offline per: {videoP}")

    # --- INIZIALIZZAZIONE RISORSE ---
    cap = cv2.VideoCapture(videoP)
    
    background = cv2.imread("back.png")
    if background is None:
        raise ValueError("ERRORE: Immagine 'back.png' non trovata!")
    bg_float = background.astype(float)
    
    print("Caricamento Rete Neurale in corso...")
    modello_pedoni = load_model('modello_pedoni.keras')
    print("Rete pronta!")
    
    CNN_INPUT_SIZE = (64, 128)
    
    # --- CONFIGURAZIONE EXPORT VIDEO (VideoWriter) ---
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Usiamo il codec mp4v che è standard per salvare in .mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter('risultato_tracking.mp4', fourcc, fps, (width, height))

    print(f"Inizio rendering video (Totale frame: {total_frames}). Attendere prego...")
    
    frame_corrente = 0

    # --- LOOP DI ELABORAZIONE OFFLINE ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_corrente += 1
        
        # Estrazione Blob
        boxes = find_blobs_static_bg(frame, bg_float, threshold=55, min_area=200)
        
        # Classificazione e Disegno
        for (x, y, w, h) in boxes:
            y_start, y_end = max(0, y), min(height, y + h)
            x_start, x_end = max(0, x), min(width, x + w)
            cut = frame[y_start:y_end, x_start:x_end]
            
            if cut.size == 0:
                continue
                
            cut_rgb = cv2.cvtColor(cut, cv2.COLOR_BGR2RGB)
            cut_resized = cv2.resize(cut_rgb, CNN_INPUT_SIZE)
            cut_batch = np.expand_dims(cut_resized, axis=0)

            predizione = modello_pedoni.predict(cut_batch, verbose=0)
            
            if predizione[0][0] < 0.5:
                # Disegna Box Verde per UMANO
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Persona {(1 - predizione[0][0])*100:.0f}%", (x, max(15, y-10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # Disegna Box Rosso per SFONDO
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
                cv2.putText(frame, "Rumore", (x, max(15, y-10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Invece di mostrare a schermo, SCRIVIAMO il frame nel file video
        out_video.write(frame)
        
        # Stampa un indicatore di progresso ogni 50 frame così capiamo che sta lavorando
        if frame_corrente % 50 == 0:
            print(f"Elaborato frame {frame_corrente}/{total_frames} ({(frame_corrente/total_frames)*100:.1f}%)")

    # --- PULIZIA FINALE ---
    cap.release()
    out_video.release() # CRITICO: se non chiami release(), il file mp4 sarà corrotto!
    print("\nRendering completato con successo!")
    print("Controlla la cartella del progetto: troverai il file 'risultato_tracking.mp4'")

if __name__ == "__main__":
    main()