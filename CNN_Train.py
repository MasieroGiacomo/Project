import matplotlib.pyplot as plt
import numpy as np
import warnings
import os
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras import Sequential
from keras.layers import Input, Rescaling, Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import image_dataset_from_directory

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Less messages in the terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

warnings.filterwarnings("ignore")

## PARAMETERS ##
IMG_HEIGHT = 128
IMG_WIDTH = 64
BATCH_SIZE = 32
EPOCHS = 10
opt = Adam(learning_rate = 1e-3)
loss_fcn = SparseCategoricalCrossentropy()
model_type = 0      # 0 = custom CNN, 1 = AlexNet


print("Loading training data...")
train_dataset = image_dataset_from_directory(
    'dataset/train', 
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='binary' 
)
print(f"The images are in the range [{np.min(train_dataset)},{np.max(train_dataset)}].")

print("loading validation data...")
val_dataset = image_dataset_from_directory(
    'dataset/validation',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='binary'
)
print(f"The images are in the range [{np.min(val_dataset)},{np.max(val_dataset)}].")

def create_model():
    if model_type == 0:                 ## CUSTOM CNN
        model = Sequential([   
            # Input layer
            Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            
            # Pixel Normalization
            Rescaling(1./255),

            # Convolutions with pooling
            Conv2D(filters = 32, kernel_size = (3, 3), activation = "relu", padding = 'same'),
            MaxPooling2D(pool_size = (2, 2)),
            Conv2D(filters = 64, kernel_size = (3, 3), activation = "relu", padding = 'same'),
            MaxPooling2D(pool_size = (2, 2)),

            # Classification head
            Flatten(), 
            Dense(128, activation = "relu"), 
            Dense(1, activation = "sigmoid")
        ])

    elif model_type == 1:               ## ALEXNET-STYLE CNN
        model = Sequential([   
            # Input layer
            Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            
            # Pixel Normalization
            Rescaling(1./255),

            # Convolutions with pooling
            Conv2D(filters = 96, kernel_size = (11, 11), activation = "relu", padding = 'same'),
            MaxPooling2D(pool_size = (3, 3)),
            Conv2D(filters = 256, kernel_size = (5, 5), activation = "relu", padding = 'same'),
            MaxPooling2D(pool_size = (3, 3)),
            Conv2D(filters = 384, kernel_size = (3, 3), activation = "relu", padding = 'same'),
            Conv2D(filters = 384, kernel_size = (3, 3), activation = "relu", padding = 'same'),
            Conv2D(filters = 256, kernel_size = (3, 3), activation = "relu", padding = 'same'),
            MaxPooling2D(pool_size = (3, 3)),

            # Classification head
            Flatten(), 
            Dense(128, activation = "relu"), 
            Dense(1, activation = "sigmoid")
        ])

    model.compile(optimizer=opt,
                  loss=loss_fcn,
                  metrics=['accuracy'])
    return model

model = create_model()
model.summary() 

print("Training...")
history = model.fit(            # For plotting
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)

## SAVING THE MODEL
model.save('Results/pedestrians.keras')
print("Training completed and model saved!")

def plot_results(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epoch = range(1, len(acc) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epoch, acc, 'b', label='Training Accuracy')
    plt.plot(epoch, val_acc, 'r', label='Validation Accuracy')
    plt.title('Accuracy di Training e Validation')
    plt.legend()
    plt.grid(True)
    
    # Loss 
    plt.subplot(1, 2, 2)
    plt.plot(epoch, loss, 'b', label='Training Loss')
    plt.plot(epoch, val_loss, 'r', label='Validation Loss')
    plt.title('Loss di Training e Validation')
    plt.legend()
    plt.grid(True)
    
    plt.show()

plot_results(history)
