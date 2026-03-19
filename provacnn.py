import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import load_model
from keras.utils import image_dataset_from_directory
import tensorflow as tf

# --- CONFIGURAZIONI ---
IMG_HEIGHT = 128
IMG_WIDTH = 64
BATCH_SIZE = 32
TEST_DIR = 'dataset_ritagliato/test' # Assicurati che il percorso sia corretto

print("Caricamento del modello...")
modello = load_model('modello_pedoni.keras')
print("Modello caricato con successo!\n")

# 1. CARICAMENTO DEL DATASET DI TEST
# REGOLA D'ORO: shuffle=False è obbligatorio per non mischiare l'ordine delle foto
# rispetto alle loro etichette originali durante la valutazione.
print("Caricamento immagini di test...")
test_dataset = image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=False 
)

nomi_classi = test_dataset.class_names # Dovrebbe essere ['sfondo', 'umani']
print(f"Classi trovate: {nomi_classi}\n")

# 2. ESTRAZIONE DELLE ETICHETTE VERE E DELLE PREDIZIONI
print("Calcolo delle predizioni su tutto il dataset di test. Attendere...")
# predict() sputa fuori i numeri da 0.0 a 1.0
predizioni_prob = modello.predict(test_dataset)

# Trasformiamo le probabilità in classi (0 o 1) con soglia 0.5
y_pred = (predizioni_prob > 0.5).astype(int).flatten()

# Estraiamo le etichette "vere" (ground truth) dal dataset
y_true = np.concatenate([y for x, y in test_dataset], axis=0).astype(int).flatten()

# 3. METRICHE E CLASSIFICATION REPORT
print("\n" + "="*50)
print("REPORT DI CLASSIFICAZIONE (PRECISION & RECALL)")
print("="*50)
print(classification_report(y_true, y_pred, target_names=nomi_classi))

# 4. CREAZIONE DELLA MATRICE DI CONFUSIONE
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
# Usiamo Seaborn per disegnare una heatmap bella e professionale
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=nomi_classi, yticklabels=nomi_classi)
plt.title('Matrice di Confusione del Modello', fontsize=16, fontweight='bold')
plt.xlabel('Predizione della Rete (Quello che ha detto il PC)', fontsize=12)
plt.ylabel('Classe Vera (Quello che era davvero)', fontsize=12)
plt.show()

# 5. VISUALIZZAZIONE DI ALCUNI ERRORI (EXTRA DA INGEGNERE)
# È sempre utile vedere "ad occhio" su quali immagini la rete si è confusa
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
    print("\nIncredibile! La rete non ha commesso NESSUN ERRORE sul test set!")