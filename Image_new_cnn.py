import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image
import coremltools as ct
from coremltools.models.neural_network import NeuralNetworkBuilder, SgdParams
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# Enhanced Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

class_names = ['me', 'not_me']

train_generator = train_datagen.flow_from_directory(
    'train_data',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    classes=class_names,
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'train_data',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    classes=class_names,
    subset='validation'
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout



# Simplified CNN Model
model = Sequential([
    Conv2D(128, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])
# Define your own learning rate
learning_rate = 0.1

# Use the learning rate in the Adam optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=30
)

def predict_image(model, image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_batch)
    predicted_class = class_names[np.argmax(predictions[0])]
    predicted_prob = np.max(predictions[0])

    return predicted_class, predicted_prob

# Example usage:
image_path = "/Users/rishabhsolanki/Desktop/Machine learning/ios/data/test.png"
predicted_class, predicted_prob = predict_image(model, image_path)
print(f"Predicted class: {predicted_class} with confidence: {predicted_prob:.2f}")

# CoreML Conversion
input_shape = (1, 150, 150, 3)
input_shape_spec = ct.Shape(shape=input_shape)
input_spec = ct.ImageType(shape=input_shape_spec, bias=[0, 0, 0], scale=1/255.0)

input_spec.name = "conv2d_input"

coreml_model = ct.convert(model, inputs=[input_spec], source="tensorflow")

layer_names1 = [layer.name for layer in coreml_model.get_spec().neuralNetwork.layers]
print(layer_names1)

# Get the names of convolutional layers from the CoreML model
conv_layer_names = [layer.name for layer in coreml_model.get_spec().neuralNetwork.layers if layer.WhichOneof('layer') == 'convolution']

# Select the name of the second convolutional layer to make it updatable
updatable_conv_layer = conv_layer_names[1]

builder = NeuralNetworkBuilder(spec=coreml_model.get_spec())
builder.add_softmax(name='output_prob', input_name=updatable_conv_layer, output_name='output_prob')

# Only making the second convolutional layer updatable
updatable_layers = [updatable_conv_layer]
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
