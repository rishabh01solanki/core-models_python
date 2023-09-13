import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Flatten, Dropout
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import coremltools as ct  # Ensure coremltools is installed

# Directory
train_data_dir = '/Users/rishabhsolanki/Desktop/Machine learning/ios/Data/train_data'

# Image dimensions
img_width, img_height = 150, 150
batch_size = 5

# ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    validation_split=0.2
)

# Training and Validation Generators
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# Model Definition
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
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# Compile Model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train Model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=2,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size
)

# Feature Extraction with CNN
feature_model = Model(inputs=model.input, outputs=model.layers[-5].output)  # Extracting features before Flatten layer
train_features = feature_model.predict(train_generator)
val_features = feature_model.predict(val_generator)

# RandomForest Training
clf = RandomForestClassifier(n_estimators=100)
clf.fit(train_features, train_generator.classes)

# Predictions
val_images, val_labels = next(val_generator)
cnn_predictions = model.predict(val_images)
val_features = feature_model.predict(val_images)
rf_predictions = clf.predict(val_features)

# Combine Predictions
final_predictions = 0.5 * cnn_predictions.flatten() + 0.5 * rf_predictions
final_class_predictions = [1 if pred > 0.5 else 0 for pred in final_predictions]

# Display Results
print("True Labels:", val_labels)
print("CNN Predictions:", cnn_predictions.flatten())
print("RandomForest Predictions:", rf_predictions)
print("Combined Predictions:", final_class_predictions)

# Visualize the results
for i in range(len(val_images)):
    plt.imshow(val_images[i])
    plt.title(f"True: {int(val_labels[i])}, CNN: {cnn_predictions[i][0]:.2f}, RF: {rf_predictions[i]}, Combined: {final_class_predictions[i]}")
    plt.show()


# Save the Keras model first
keras_model_path = './model.h5'
model.save(keras_model_path)

# Print the names of the last 3 layers
for layer in model.layers[-3:]:
    print("Layer Name:", layer.name, "Layer Type:", type(layer).__name__)

# Convert to CoreML
def convert_keras_to_mlmodel(keras_path, mlmodel_path):
    keras_model = load_model(keras_path)
    mlmodel = ct.convert(keras_model)
    mlmodel.save(mlmodel_path)


import coremltools
from coremltools.models.neural_network import SgdParams
from keras.models import load_model


# Convert to CoreML
coreml_model_path = './MyModel.mlmodel'
model.save(keras_model_path)
coreml_model = ct.convert(model)
coreml_model.save(coreml_model_path)
print("CoreML Model saved!")



def make_updatable(mlmodel_url, mlmodel_updatable_path):
    # Load the model
    spec = coremltools.utils.load_spec(mlmodel_url)
    builder = coremltools.models.neural_network.NeuralNetworkBuilder(spec=spec)

    # Inspect the last few layers
    print(builder.inspect_layers(last=5))

    # Make the last two dense layers updatable
    builder.make_updatable(['sequential/dense_2/BiasAdd', 'sequential/dense_1/BiasAdd'])  # Adjust names if they differ in your model

    # Retrieve name of the output of the last layer
    last_layer_output = builder.spec.neuralNetwork.layers[-1].output[0]

    # Add the softmax layer
    builder.add_softmax(name='SoftmaxLayer', input_name=last_layer_output, output_name='output1')


    # Set the loss and optimizer
    builder.set_categorical_cross_entropy_loss(name='lossLayer', input='output1')  # 'output1' is the output of the softmax layer
    builder.set_sgd_optimizer(SgdParams(lr=0.01, batch=1))
    builder.set_epochs(5)

    # Set training input descriptions
    spec.description.trainingInput[0].shortDescription = 'Input image for training'
    spec.description.trainingInput[1].shortDescription = 'True label for the input image'
    
    # Save the updated model
    updatable_model = coremltools.models.MLModel(spec)
    updatable_model.save(mlmodel_updatable_path)


coreml_updatable_model_path = './UpdatableModel.mlmodel'
make_updatable(coreml_model_path, coreml_updatable_model_path)

# Verify the model's updatable layers, loss, and optimizer
builder = coremltools.models.neural_network.NeuralNetworkBuilder(spec=coremltools.utils.load_spec(coreml_updatable_model_path))
print("Loss Layers:", builder.inspect_loss_layers())
print("Optimizer:", builder.inspect_optimizer())
print("Updatable Layers:", builder.inspect_updatable_layers())



    
