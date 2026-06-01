import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = 'train'
test_dir = 'test'

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=10, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(train_dir, target_size=(48,48), color_mode='grayscale', batch_size=64, class_mode='categorical')
test_data = test_datagen.flow_from_directory(test_dir, target_size=(48,48), color_mode='grayscale', batch_size=64, class_mode='categorical')

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    BatchNormalization(), MaxPooling2D(2,2), Dropout(0.25),
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(), MaxPooling2D(2,2), Dropout(0.25),
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(), MaxPooling2D(2,2), Dropout(0.25),
    Flatten(),
    Dense(256, activation='relu'), Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, validation_data=test_data, epochs=10)
model.save('emotion_model.h5')
print("Model saved!")