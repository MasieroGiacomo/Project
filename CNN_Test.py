import os
os.environ["KERAS_BACKEND"] = "torch"
import matplotlib.pyplot as plt
import numpy as np
import warnings
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import load_model
from keras.utils import image_dataset_from_directory

# Less messages in the terminal
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

## PARAMETERS ##
IMG_HEIGHT = 128
IMG_WIDTH = 64
BATCH_SIZE = 32
model_type = 'steve'      

# ================================================ #
### 1. LOADING OF MODEL AND TEST DATASET

if model_type == 'steve':
    model = load_model('Models/steve.keras')
elif model_type == 'alex':       
    model = load_model('Models/alex.keras')
elif model_type == 'vgg':
    model = load_model('Models/vgg.keras')
print("Model loading...")

print("Loading test images...")
test_dataset = image_dataset_from_directory(
    'dataset/test' ,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=False               # To preserve the order of files and labels for correct evaluation
)

classes = test_dataset.class_names 
print(f"Classes found: {classes}\n")

# ================================================ #
### 2. LABELS EXTRACTION AND PREDICTIONS
print("Testing...")
prediction_prob = model.predict(test_dataset)     # predict() gives numbers from 0.0 to 1.0

# From probabilities to classes 
y_pred = (prediction_prob > 0.5).astype(int).flatten()

# Getting real labels (y_true) from the dataset
y_true = np.concatenate([y for x, y in test_dataset], axis=0).astype(int).flatten()

# ================================================ #
### 3. REPORT 
print("\n" + "="*50)
print("PRECISION & RECALL")
print("="*50)
print(classification_report(y_true, y_pred, target_names=classes))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.xlabel('Model Prediction', fontsize=12)
plt.ylabel('True Class', fontsize=12)

if model_type == 'steve':
        plt.savefig('Results/test_plot_steve.png')
elif model_type == 'alex':       
        plt.savefig('Results/test_plot_alexpng')
elif model_type == 'vgg':
        plt.savefig('Results/test_plot_vgg.png')
plt.show()

'''# 5. VISUALIZZAZIONE DI ALCUNI ERRORI (EXTRA DA INGEGNERE)
errori_idx = np.where(y_pred != y_true)[0]

if len(errori_idx) > 0:
    print(f"\nLa rete ha commesso {len(errori_idx)} errori su {len(y_true)} immagini totali.")
    print("Visualizzazione di alcuni degli errori...")
    
    # Prendi fino a 4 immagini sbagliate a caso
    num_errori_da_mostrare = min(4, len(errori_idx))
    indici_scelti = np.random.choice(errori_idx, num_errori_da_mostrare, replace=False)
    
    # Ricarichiamo i file per mostrarli (test_dataset.file_paths contiene tutti i nomi file in ordine)
    paths = test_dataset.file_paths
    
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(indici_scelti):
        img_path = paths[idx]
        img = plt.imread(img_path)
        
        classe_vera = nomi_classi[y_true[idx]]
        classe_predetta = nomi_classi[y_pred[idx]]
        confidenza = predizioni_prob[idx][0]
        if y_pred[idx] == 0:
            confidenza = 1.0 - confidenza # Invertiamo la probabilità se ha predetto sfondo
            
        plt.subplot(1, num_errori_da_mostrare, i+1)
        plt.imshow(img)
        plt.title(f"Vero: {classe_vera}\nPredetto: {classe_predetta} ({confidenza*100:.1f}%)", color='red')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
else:
    print("\nIncredibile! La rete non ha commesso NESSUN ERRORE sul test set!")'''