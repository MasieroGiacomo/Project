from keras.applications import VGG16
from keras.layers import Input, Rescaling, Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras import Sequential
from keras import Model
from keras.utils import image_dataset_from_directory
import numpy as np

IMG_HEIGHT = 128
IMG_WIDTH = 64
BATCH_SIZE = 32
EPOCHS = 10

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
base_model.summary()
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

lr=1e-4 #learning rate
model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])

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


print("Training...")
history = model.fit(            # For plotting
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)

## SAVING THE MODEL
model.save('Models/transfer_vgg.keras')
print("Training completed and model saved!")