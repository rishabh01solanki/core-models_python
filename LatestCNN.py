import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

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
    batch_size=5,
    class_mode='categorical',
    classes=class_names,
    subset='training',
    color_mode='grayscale'
)

validation_generator = train_datagen.flow_from_directory(
    'train_data',
    target_size=(150, 150),
    batch_size=5,
    class_mode='categorical',
    classes=class_names,
    subset='validation',
    color_mode='grayscale'
)

# Simplified CNN Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation

# Define layers separately
input_conv = Conv2D(16, (3, 3), use_bias=False, input_shape=(150, 150, 1), kernel_initializer='uniform',name='conv2d_input')
input_activation = Activation('relu')
pooling_layer1 = MaxPooling2D(2, 2)

conv_layer2 = Conv2D(32, (3, 3), use_bias=False, kernel_initializer='uniform')
activation2 = Activation('relu')
pooling_layer2 = MaxPooling2D(2, 2)

conv_layer3 = Conv2D(64, (3, 3), use_bias=False, kernel_initializer='uniform')
activation3 = Activation('relu')
pooling_layer3 = MaxPooling2D(2, 2)

conv_layer4 = Conv2D(128, (3, 3), use_bias=False, kernel_initializer='uniform')
activation4 = Activation('relu')
pooling_layer4 = MaxPooling2D(2, 2)

flatten_layer = Flatten()
dense_layer1 = Dense(64, activation='relu', kernel_initializer='uniform')
dropout_layer = Dropout(0.2)
output_layer = Dense(len(class_names), activation='softmax')

# Construct the model
model = Sequential([
    input_conv, 
    input_activation,
    pooling_layer1,
    
    conv_layer2,
    activation2,
    pooling_layer2,
    
    conv_layer3,
    activation3,
    pooling_layer3,
    
    conv_layer4,
    activation4,
    pooling_layer4,
    
    flatten_layer,
    dense_layer1,
    dropout_layer,
    output_layer
])


# Use the learning rate in the Adam optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=2
)


# CoreML Conversion
import coremltools as ct
from coremltools.models.neural_network import NeuralNetworkBuilder, SgdParams

input_shape = (1, 150, 150, 1)
input_shape_spec = ct.Shape(shape=input_shape)
input_spec = ct.ImageType(shape=input_shape_spec, bias=[0,0,0], scale=1/255.0)

input_spec.name = "conv2d_input"

# Convert the model to CoreML format
coreml_model = ct.convert(model, inputs=[input_spec], source="tensorflow")

# Define the convolutional layers that you want to make updatable
updatable_conv_layers = ['sequential/conv2d_3/Conv2Dx', 'sequential/conv2d_2/Conv2Dx']

# Use the NeuralNetworkBuilder to modify the model
builder = NeuralNetworkBuilder(spec=coreml_model.get_spec())

# As we identified, the last layer's output name is 'Identity'
last_layer_output_name = 'Identity'

# Add the softmax layer connected to the last layer's output
builder.add_softmax(name='output_prob', input_name=last_layer_output_name, output_name='output_prob')

# Connect the softmax output to the loss
builder.set_categorical_cross_entropy_loss(name='LossLayer', input='output_prob')

# Make the defined convolutional layers updatable
builder.make_updatable(updatable_conv_layers)

# Set optimizer and epochs
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
