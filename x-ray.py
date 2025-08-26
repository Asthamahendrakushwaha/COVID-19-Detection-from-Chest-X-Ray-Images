import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Paths to dataset (after unzipping Kaggle dataset)
train_dir = "C:\\astha\\ml-based-project\\project-02\\COVID-19_Radiography_Dataset\\train"
test_dir  = "C:\\astha\\ml-based-project\\project-02\\COVID-19_Radiography_Dataset\\test"


# Data preprocessing & augmentation
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=15, zoom_range=0.2,
                                   width_shift_range=0.1, height_shift_range=0.1,
                                   horizontal_flip=True, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(150, 150), batch_size=32,
    class_mode='binary', subset='training')

val_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(150, 150), batch_size=32,
    class_mode='binary', subset='validation')

# CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training
history = model.fit(train_generator, epochs=10, validation_data=val_generator)

# Plot training results
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

# Save model
model.save("covid_xray_classifier.h5")
