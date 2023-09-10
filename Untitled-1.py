# %%
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


# %%
# Directory
train_data_dir = '/Users/rishabhsolanki/Desktop/Machine learning/ios/Data/train_data'
# Image dimensions
img_width, img_height = 150, 150
batch_size = 5  # Adjust this as per the computation capacity of your system


# %%
# Initialize ImageDataGenerator with validation split
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    validation_split=0.2  # Using 20% of the data for validation
)

# Configure the train_generator for training data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'  # Specify this subset for training data
)

# Configure the val_generator for validation data
val_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'  # Specify this subset for validation data
)

# %%
# Model definition
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# %%
from keras.optimizers import Adam  # or any other optimizer you want to use

# Define the learning rate
learning_rate = 0.01

# Instantiate the optimizer with the desired learning rate
optimizer = Adam(learning_rate=learning_rate)

# Now, use this optimizer while compiling the model
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


# %%
# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=20,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size
)

# %%
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential, Model

# %%

# Extract features using the CNN
feature_model = Model(inputs=model.input, outputs=model.layers[-4].output)  # Extracting features before Flatten layer
train_features = feature_model.predict(train_generator)
val_features = feature_model.predict(val_generator)



# %%
# Train a RandomForest using these features
clf = RandomForestClassifier(n_estimators=100)
clf.fit(train_features, train_generator.classes)


# %%
# Get the images and labels from the validation generator
val_images, val_labels = next(val_generator)

# Predict using CNN
cnn_predictions = model.predict(val_images)

# Extract features from these images using the CNN
val_features = feature_model.predict(val_images)

# Predict using RandomForest
rf_predictions = clf.predict(val_features)

# %%
# Combining predictions with equal weightage
final_predictions = 0.5 * cnn_predictions.flatten() + 0.5 * rf_predictions

# Convert combined predictions to binary class labels
final_class_predictions = [1 if pred > 0.5 else 0 for pred in final_predictions]


# %%
# Print the results
print("True Labels:", val_labels)
print("CNN Predictions:", cnn_predictions.flatten())
print("RandomForest Predictions:", rf_predictions)
print("Combined Predictions:", final_class_predictions)

# %%
import matplotlib.pyplot as plt

# Visualize the results
for i in range(len(val_images)):
    plt.imshow(val_images[i])
    plt.title(f"True: {int(val_labels[i])}, CNN: {cnn_predictions[i][0]:.2f}, RF: {rf_predictions[i]}, Combined: {final_class_predictions[i]}")
    plt.show()



