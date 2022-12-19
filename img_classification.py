#%%
#import packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,optimizers,losses,callbacks,applications
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import TensorBoard
import datetime
import numpy as np
import os
import matplotlib.pyplot as plt 
# %%

path_to_file = os.path.join(os.getcwd(),'dataset')

#Define batch size and image size
BATCH_SIZE = 32
IMG_SIZE = (160,160)
# %%
#Load the data as tensorflow dataset using the special method
training_dataset = keras.utils.image_dataset_from_directory(path_to_file,validation_split=0.2,subset="training",batch_size=BATCH_SIZE,image_size=IMG_SIZE,shuffle=True,seed=12345) 
val_dataset=keras.utils.image_dataset_from_directory (path_to_file,validation_split=0.2,subset="validation",batch_size=BATCH_SIZE,image_size=IMG_SIZE,shuffle=True,seed=12345)

#%%
#Batches dataset
val_batches = tf.data.experimental.cardinality(val_dataset)
test_dataset = val_dataset.take(val_batches // 2)
val_dataset = val_dataset.skip(val_batches // 2)


#%%

print('Number of validation batches: %d' % tf.data.experimental.cardinality(val_dataset))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))
#%%
#Class names
class_names = training_dataset.class_names
#%%
# Plot some examples
plt.figure(figsize=(10,10))
for images,labels in training_dataset.take(1):
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')


#%%
#5 Convert the Batch dataset into Prefetch dataset

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = training_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)


#%%
# Create a 'model' for image augmentation
data_augmentation = keras.Sequential()
data_augmentation.add(layers.RandomFlip('horizontal'))
data_augmentation.add(layers.RandomRotation(0.2))

#%%
#Repeatedly apply data augmentation on one image and see the result

for images,labels in train_dataset.take(1):
    first_image = images[0]
    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3,3,i+1)
        augmented_image = data_augmentation(tf.expand_dims(first_image,axis=0))
        plt.imshow(augmented_image[0]/255.0)
        plt.axis('off')

#%%
#Before transfer learning 
#Create the layer for data normalization

preprocess_input = applications.mobilenet_v2.preprocess_input

# %%
#Start the transfer learning 
#Instantiate the pretrained model
IMG_SHAPE = IMG_SIZE + (3,)
base_model = applications.MobileNetV2(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')

# Set the trainable model as non-trainabale(frozen)
base_model.trainable = False
base_model.summary()
keras.utils.plot_model(base_model,show_shapes=True)

# %%
#Create the classifier
#Create the global average pooling layer
global_avg = layers.GlobalAveragePooling2D()

#Create an output layer
output_layer = layers.Dense(len(class_names),activation='softmax')

#Link the layers together to form a pipeline with functional API
inputs = keras.Input(shape=IMG_SHAPE)

x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x,training=False)
x = global_avg(x)
x = layers.Dropout(0.3)(x)
outputs = output_layer(x)

#Instantiate the full model pipeline
model = keras.Model(inputs=inputs,outputs=outputs)
print(model.summary())

#Compile the model
optimizer = optimizers.Adam(learning_rate=0.0001)
loss = losses.SparseCategoricalCrossentropy()
model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])

#13 Evaluate the model before the training

loss0, acc0 = model.evaluate(test_dataset)

print("----------------------------Evaluation Before Training")
print("Loss =",loss0)
print("Accuracy =",acc0)

# %%

LOGS_PATH = os.path.join(os.getcwd(),'logs', datetime.datetime.now().strftime('%Y&m%d-%H%M%S'))
tensorboard_callback = TensorBoard(log_dir=LOGS_PATH)

#%%
#Run on 10 epochs
EPOCHS = 10
history = model.fit(training_dataset,validation_data=test_dataset,epochs=EPOCHS,callbacks=[tensorboard_callback])

#%%
#Evaluate the model after training

test_loss,test_acc = model.evaluate(test_dataset)
print("---------------------------------------Evaluation After Training-----------------")
print("Loss = ",test_loss)
print("Accuracy = ",test_acc)

#%%
# Use the model to perform prediction

image_batch, label_batch = test_dataset.as_numpy_iterator().next()
y_pred = np.argmax(model.predict(image_batch),axis=1)

#Model Evaluation (Classification report)

print("Classification report :\n", classification_report(label_batch,y_pred))

#%%
model.save(filepath=os.path.join(os.getcwd(), 'saved_model','text.h5'))