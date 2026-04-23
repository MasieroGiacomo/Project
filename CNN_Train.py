import matplotlib.pyplot as plt
import numpy as np
import warnings
import os
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras import Sequential
from keras.layers import Input, Rescaling, Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.utils import image_dataset_from_directory

# Less messages in the terminal
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

## PARAMETERS ##
IMG_HEIGHT = 128
IMG_WIDTH = 64
BATCH_SIZE = 32
EPOCHS = 20
opt = Adam(learning_rate = 1e-3)
loss_fcn = BinaryCrossentropy()
model_type = 'vgg'       # 'steve' -> custom CNN, 'alex' -> AlexNet-like, 'vgg' -> VGG-like 

if model_type == 'steve':
    model_path = 'Models/pedestrians_steve.keras'
    save_path = 'Results/training_plot_steve.png'       
elif model_type == 'alex':
    model_path = 'Models/pedestrians_alex2.keras' 
    save_path = 'Results/training_plot_alex.png'    
elif model_type == 'vgg':
    model_path = 'Models/pedestrians_vgg2.keras'
    save_path = 'Results/training_plot_vgg2.png'


# ================================================ #
### 1. LOADING OF THE DATASETS
print("Loading training data...")
train_dataset = image_dataset_from_directory(
    'dataset/train', 
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='binary' 
)

print("loading validation data...")
val_dataset = image_dataset_from_directory(
    'dataset/validation',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

# ================================================ #
### 2. MODEL CREATION

def create_model():
    if model_type == 'steve':       ## TRAINING TIME : 6 min        
        model = Sequential([   
            # Input layer
            Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),    # img = 128x64x3
            
            # Pixel Normalization
            Rescaling(1./255),

            # Convolutions with pooling
            Conv2D(filters = 32, kernel_size = (3, 3), activation = "relu", padding = 'same'),  
            MaxPooling2D(pool_size = (2, 2)),   # img -> 64x32x32
            Conv2D(filters = 64, kernel_size = (3, 3), activation = "relu", padding = 'same'),
            MaxPooling2D(pool_size = (2, 2)),   # img -> 32x16x64

            # Classification head
            Flatten(), 
            Dense(128, activation = "relu"), 
            Dense(1, activation = "sigmoid")        # Binary classification 
        ])

    elif model_type == 'alex':       ## TRAINING TIME : 40 min  /   
        model = Sequential([   
            # Input layer
            Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),    # img = 128x64x3
            
            # Pixel Normalization
            Rescaling(1./255),

            # Convolutions with pooling
            Conv2D(filters = 96, kernel_size = (9, 9), activation = "relu", strides = 2, padding = 'same'),  # img -> 64x32x96
            MaxPooling2D(pool_size = (3, 3), strides = 2, padding = 'valid'),    # valid = no padding, img -> 31x15x96

            Conv2D(filters = 256, kernel_size = (5, 5), activation = "relu", strides = 1, padding = 'same'),
            MaxPooling2D(pool_size = (3, 3), strides = 2, padding = 'valid'),    # img -> 15x7x256

            Conv2D(filters = 384, kernel_size = (3, 3), activation = "relu", padding = 'same'),
            Conv2D(filters = 384, kernel_size = (3, 3), activation = "relu", padding = 'same'),
            Conv2D(filters = 256, kernel_size = (3, 3), activation = "relu", padding = 'same'),
            MaxPooling2D(pool_size = (3, 3), strides = 2, padding = 'valid'),    # img -> 7x3x256

            # Classification head
            # The Dropout layers are used to reduce overfitting,by randomly set a fraction of the input units to 0 during training, 
            # which helps the model to generalize better.
            Flatten(), 
            Dense(256, activation = "relu"),    # OG 128
            Dropout(0.5),                      
            Dense(256, activation = "relu"),    # OG 128
            Dropout(0.5),
            Dense(1, activation = "sigmoid")
        ])

    elif model_type == 'vgg':    ## TRAINING TIME : 15 min
        model = Sequential([
            # Input layer
            Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),    # img = 128x64x3

            # Pixel Normalization
            Rescaling(1./255),

            # Convolutions with pooling
            Conv2D(filters = 32, kernel_size = (3, 3), activation = "relu", strides = 1, padding = 'same'),
            Conv2D(filters = 32, kernel_size = (3, 3), activation = "relu", strides = 1,padding = 'same'),
            MaxPooling2D(pool_size = (2, 2)),   # stride = 2 by default, no padding, img -> 64x32x32

            Conv2D(filters = 64, kernel_size = (3, 3), activation = "relu", strides = 1,padding = 'same'),
            Conv2D(filters = 64, kernel_size = (3, 3), activation = "relu", strides =1,padding = 'same'),
            MaxPooling2D(pool_size = (2, 2)),   # img -> 32x16x64

            Conv2D(filters = 128, kernel_size = (3, 3), activation = "relu", strides = 1,padding = 'same'),
            Conv2D(filters = 128, kernel_size = (3, 3), activation = "relu", strides = 1,padding = 'same'),
            MaxPooling2D(pool_size = (2, 2)),   # img -> 16x8x128

            # Classification head
            Flatten(), 
            Dense(512, activation = "relu"),    # OG 128
            Dropout(0.5),
            Dense(1, activation = "sigmoid")
        ])

    model.compile(loss=loss_fcn,
                  optimizer=opt,
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

# ================================================ #
### 3. MODEL SAVING
model.save(model_path)
print("Training completed and model saved!")

# ================================================ #
### 4. REPORT
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
    
plt.savefig(save_path)
plt.show()

'''## FROM LAB 6
plt.figure(figsize=(20, 3))
for i, metric in enumerate(["accuracy", "loss"]):
    plt.subplot(1, 2, i + 1) 
    plt.plot(model.history.history[metric])
    plt.plot(model.history.history["val_" + metric])
    plt.title("Model {}".format(metric))
    plt.xlabel("epochs")
    plt.ylabel(metric)
    plt.legend(["train", "val"])'''