# %%
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Flatten, Dropout
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import coremltools as ct  # Ensure coremltools is installed


# %%
# Directory
train_data_dir = '/Users/rishabhsolanki/Desktop/Machine learning/ios/Data/train_data'

# Image dimensions
img_width, img_height = 150, 150
batch_size = 5


# %%
# ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
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


# %%
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout

# Create a custom input tensor
input_tensor = Input(shape=(img_width, img_height, 3), name='conv2d_input')

# Add MobileNetV2 as base model using custom input tensor
base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=input_tensor)

# Define a modified simplified CNN model to add on top
model_top = Sequential([
    Conv2D(4, (1, 1), use_bias=False, kernel_initializer='uniform', activation='relu', input_shape=base_model.output_shape[1:]),
    MaxPooling2D(1, 1),
    Flatten(),
    Dense(4, activation='relu', kernel_initializer='uniform'),
    Dropout(0.2),
    Dense(1, activation='sigmoid') # use 'softmax' for multi-class and adjust the units accordingly
])

# Combine MobileNetV2 output with the top model
predictions = model_top(base_model.output)

# Combine the input tensor and final output layer into the final model
model = Model(inputs=input_tensor, outputs=predictions)

# Freeze the layers of the MobileNetV2 model
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# %%
# Train Model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=8,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size
)

# %%
# Predictions
val_images, val_labels = next(val_generator)
cnn_predictions = model.predict(val_images)


# %%
# Display Results
print("True Labels:", val_labels)
print("CNN Predictions:", cnn_predictions.flatten())

# %%
# Visualize the results
for i in range(len(val_images)):
    plt.imshow(val_images[i])
    plt.title(f"True: {int(val_labels[i])}, CNN: {cnn_predictions[i][0]:.2f}")
    plt.show()

# %%
import coremltools
from coremltools.models.neural_network import NeuralNetworkBuilder, SgdParams

# %%
# Save the Keras model first
keras_model_path = './model.h5'
model.save(keras_model_path)

# %%
# If the model expects 3 channels, adjust the input shape and preprocessing details
input_shape = (1, 150, 150, 3)
input_shape_spec = ct.Shape(shape=input_shape)
input_spec = ct.ImageType(shape=input_shape_spec, bias=[0, 0, 0], scale=1/255.0)
input_spec.name = "conv2d_input"


# %%
# Convert the Keras model to CoreML model with specified input details
coreml_model_path = './MyModel.mlmodel'
coreml_model = ct.convert(model, inputs=[input_spec], source="tensorflow")
coreml_model.save(coreml_model_path)
print("CoreML Model saved!")


# %%
def make_updatable(mlmodel_url, mlmodel_updatable_path):
    # Load the model
    spec = ct.utils.load_spec(mlmodel_url)
    builder = ct.models.neural_network.NeuralNetworkBuilder(spec=spec)

    # Inspect the last few layers
    print(builder.inspect_layers(last=5))

    # Make the last two dense layers updatable
    builder.make_updatable(['model_4/sequential_4/dense_8/BiasAdd' , 'model_4/sequential_4/dense_9/BiasAdd'])  # Adjust names if they differ in your model

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
    updatable_model = ct.models.MLModel(spec)
    updatable_model.save(mlmodel_updatable_path)

# %%
coreml_updatable_model_path = './UpdatableModel.mlmodel'
make_updatable(coreml_model_path, coreml_updatable_model_path)

# %%
# Verify the model's updatable layers, loss, and optimizer
builder = coremltools.models.neural_network.NeuralNetworkBuilder(spec=coremltools.utils.load_spec(coreml_updatable_model_path))
builder.inspect_loss_layers()
builder.inspect_optimizer()
# let's see which layers are updatable
builder.inspect_updatable_layers()



