import matplotlib.pyplot as plt
import numpy as np
import warnings
import os

# Disattiva le ottimizzazioni custom oneDNN (come suggerito dal tuo terminale)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Imposta il livello minimo di log in C++ di TensorFlow
# 0 = tutti i log (default), 1 = nascondi INFO, 2 = nascondi INFO e WARNING, 3 = nascondi INFO, WARNING e ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- IMPORTAZIONI DIRETTE DA KERAS ---
from keras import Sequential
from keras.layers import Input, Rescaling, Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import image_dataset_from_directory

warnings.filterwarnings("ignore")

# Dimensioni standard per i pedoni
IMG_HEIGHT = 128
IMG_WIDTH = 64
BATCH_SIZE = 32

print("Caricamento dataset di Training...")
# Usiamo direttamente la funzione di Keras senza passare per tf
train_dataset = image_dataset_from_directory(
    'dataset/train', 
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='binary' # 'binary' perché abbiamo solo 2 classi (umano / non umano)
)
print(f"The images are in the range [{np.min(train_dataset)},{np.max(train_dataset)}].")

print("Caricamento dataset di Test/Validation...")
val_dataset = image_dataset_from_directory(
    'dataset/validation',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='binary'
)
print(f"The images are in the range [{np.min(val_dataset)},{np.max(val_dataset)}].")

def crea_modello():
    model = Sequential([   
        # Input layer.
        Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        
        # NORMALIZZAZIONE DEI PIXEL (fondamentale!)
        Rescaling(1./255),

        # Convolutions with subsequent pooling.
        Conv2D(filters = 32, kernel_size = (3, 3), activation = "relu", padding = 'same'),
        MaxPooling2D(pool_size = (2, 2)),
        Conv2D(filters = 64, kernel_size = (3, 3), activation = "relu", padding = 'same'),
        MaxPooling2D(pool_size = (2, 2)),

        # Classification head.
        Flatten(), # <-- PRIMA appiattiamo
        Dense(128, activation = "relu"), # <-- POI il layer denso
        Dense(1, activation = "sigmoid")
    ])

    # Compilazione del modello
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

modello = crea_modello()
modello.summary() # Stampa a schermo l'architettura della tua rete

EPOCHS = 10

print("Inizio Addestramento...")
# Salviamo la "storia" dell'addestramento nella variabile history per fare i grafici dopo
history = modello.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)

# Salva il modello sul disco in formato nativo Keras v3 (.keras)
modello.save('modello_pedoni.keras')
print("Addestramento completato e modello salvato!")

def plot_risultati(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epoche = range(1, len(acc) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Grafico Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epoche, acc, 'b', label='Training Accuracy')
    plt.plot(epoche, val_acc, 'r', label='Validation Accuracy')
    plt.title('Accuracy di Training e Validation')
    plt.legend()
    plt.grid(True)
    
    # Grafico Loss (Errore)
    plt.subplot(1, 2, 2)
    plt.plot(epoche, loss, 'b', label='Training Loss')
    plt.plot(epoche, val_loss, 'r', label='Validation Loss')
    plt.title('Loss di Training e Validation')
    plt.legend()
    plt.grid(True)
    
    plt.show()

plot_risultati(history)
