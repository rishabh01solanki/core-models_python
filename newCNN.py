
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# Setting the paths
train_data_path = '/Users/rishabhsolanki/Desktop/Machine learning/ios/Data/train_data'

# Set up parameters
img_width, img_height = 150, 150
epochs = 15
batch_size = 8

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2)  # Adding grayscale mode

train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='training',
    color_mode='grayscale')  # Adding grayscale mode

validation_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    color_mode='grayscale')  # Adding grayscale mode


model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(img_width, img_height, 1), kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 2nd Convolutional Layer
model.add(Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), activation='relu', kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 3rd Convolutional Layer
model.add(Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), activation='relu', kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten
model.add(Flatten())

# Fully Connected Layer
model.add(Dense(units=128, activation='relu', kernel_initializer='uniform'))

# Output Layer
model.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))

# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()


# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size)

# CoreML Conversion

import coremltools as ct
from coremltools.models.neural_network import NeuralNetworkBuilder, SgdParams

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
