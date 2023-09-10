import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Dense, Dropout

# Enhanced Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    shear_range=0.10,
    zoom_range=0.10,
    horizontal_flip=True
)

class_names = ['me', 'not_me']

train_generator = train_datagen.flow_from_directory(
    'train_data',
    target_size=(150, 150),
    batch_size=8,
    class_mode='categorical',
    classes=class_names,
    subset='training',
    color_mode='grayscale'
)

validation_generator = train_datagen.flow_from_directory(
    'train_data',
    target_size=(150, 150),
    batch_size=8,
    class_mode='categorical',
    classes=class_names,
    subset='validation',
    color_mode='grayscale'
)

# Using MobileNetV2 as the base model
input_tensor = tf.keras.Input(shape=(150, 150, 1))
base_model = tf.keras.applications.MobileNetV2(input_shape=(150, 150, 1), include_top=False, weights=None, input_tensor=input_tensor)

base_model.trainable = True

# Model definition with custom convolutional layers
# Model definition with custom convolutional layers
model = Sequential([
    base_model,
    Conv2D(32, (3, 3), use_bias=False),  # First custom convolutional layer without activation
    tf.keras.layers.Activation('relu'), # Separate activation
    Conv2D(32, (3, 3), use_bias=False),  # Second custom convolutional layer without activation
    tf.keras.layers.Activation('relu'), # Separate activation
    GlobalAveragePooling2D(),
    Dense(64, activation='relu', kernel_initializer='uniform', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])


# Compile and Train
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)

# CoreML Conversion
import coremltools as ct
from coremltools.models.neural_network import NeuralNetworkBuilder, SgdParams

input_shape_spec = ct.Shape(shape=(1, 150, 150, 1))
input_spec = ct.ImageType(shape=input_shape_spec, bias=[0,0,0], scale=1/255.0)

coreml_model = ct.convert(model, inputs=[input_spec], source="tensorflow")

layer_names = [layer.name for layer in coreml_model.get_spec().neuralNetwork.layers]
for idx, name in enumerate(layer_names):
    print(idx, name)


# Assuming layer_names contains the names of all the layers in your CoreML model:
updatable_layer1 = 'sequential/conv2d/Conv2Dx' # Corresponding to first Conv2D layer you added
updatable_layer2 = 'sequential/conv2d_1/Conv2Dx' # Corresponding to second Conv2D layer you added

# Combine them into a list
updatable_layers = [updatable_layer1, updatable_layer2]

builder = NeuralNetworkBuilder(spec=coreml_model.get_spec())
builder.add_softmax(name='output_prob', input_name=updatable_layer2, output_name='output_prob')


# Mark both convolutional layers as updatable
builder.make_updatable(updatable_layers)


builder.set_categorical_cross_entropy_loss(name='lossLayer', input='output_prob')
builder.set_sgd_optimizer(SgdParams(lr=0.01, batch=1))
builder.set_epochs(10)

builder.spec.description.input[0].shortDescription = 'Input image to classify'
builder.spec.description.output[0].shortDescription = 'Predicted class label/Score'
builder.spec.description.metadata.author = 'Rishabh Solanki'
builder.spec.description.metadata.license = 'Use wisely'
builder.spec.description.metadata.shortDescription = 'A custom CNN model for image classification that can be fine-tuned.'

updatable_coreml_model = ct.models.MLModel(builder.spec)
updatable_coreml_model.save("new_custom_cnn_updatable.mlmodel")

print("Updated Custom CNN-based CoreML Model saved!")
