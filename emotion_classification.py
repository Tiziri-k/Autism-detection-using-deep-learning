from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import cv2
from keras.models import Sequential
from keras.layers import Dropout,  Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
import datetime ,zipfile
import matplotlib.pyplot as plt


# import zipfile
# with zipfile.ZipFile("dataset0.zip","r") as zip_ref:
#     zip_ref.extractall("dataset0/")

# def delete_augmented_images(folder_path, prefix='aug_'):
#     all_files = os.listdir(folder_path)
#     augmented_files = [f for f in all_files if f.startswith(prefix)]
#     for file_name in augmented_files:
#         file_path = os.path.join(folder_path, file_name)
#         os.remove(file_path)

#     print(f"Deletion of augmented images completed in folder: {folder_path}")

# delete_augmented_images('dataset02/anger')
# delete_augmented_images('dataset02/fear')
# delete_augmented_images('dataset02/joy')
# delete_augmented_images('dataset02/surprise')
# delete_augmented_images('dataset02/sad')



def count_images_in_folder(folder_path):
    try:
        files = os.listdir(folder_path)

        num_images = len([file for file in files if file.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))])
        if num_images is not None:
                print(f"Number of images in '{folder_path}': {num_images}")
        else:
                print("Failed to count images.")
    except Exception as e:
        print(f"An error occurred: {e}")



num_images_fear = count_images_in_folder('dataset0/fear')
num_images_joy = count_images_in_folder('dataset0/joy')
num_images_sad = count_images_in_folder('dataset0/sad')
num_images_surprise = count_images_in_folder('dataset0/surprise')
num_images_anger = count_images_in_folder('dataset0/anger')
print("Number of images in the fear folder {}\nNumber of images in the joy folder {}\nNumber of images in the sad folder {}\nNumber of images in the sad folder {} \nNumber of images in the surprise folder {}".format(num_images_fear,num_images_joy,num_images_sad,num_images_surprise,num_images_anger))


# test it 
img = cv2.imread('dataset0//anger/02.jpg')
print(img.shape)


data_dir = 'dataset0'


datagen_kwargs = dict(rescale=1./255, validation_split=.12)
dataflow_kwargs = dict(target_size=(250, 250),
                       batch_size=16)

valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    **datagen_kwargs)
valid_generator = valid_datagen.flow_from_directory(
    data_dir, subset="validation", shuffle=False, **dataflow_kwargs)


train_datagen =  tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    **datagen_kwargs
)

train_generator = train_datagen.flow_from_directory(
    data_dir, subset="training", shuffle=False, **dataflow_kwargs)

print("Number of classes:", len(train_generator.class_indices))
print("Class labels:", train_generator.class_indices)
print("Number of training samples:", train_generator.samples)
print("Number of validation samples:", valid_generator.samples)

images, labels = next(train_generator)
print("Shape of the first batch of images:", images.shape)
print("Labels of the first batch of images:", labels)


plt.imshow(images[0])
plt.show()


def create_model(num_classes=5):
    feature_extractor_layer = hub.KerasLayer(
        "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2",
        trainable=False,  
        name='feature_extraction_layer',
        input_shape=(250, 250, 3)  
    )

    model = Sequential([
        feature_extractor_layer,
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax', name='output_layer')
    ])
    return model

def create_tensorboard_callback(dir_name, experiment_name):
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir)
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback

data_dir = 'dataset0'
datagen_kwargs = dict(rescale=1./255, validation_split=.10)
dataflow_kwargs = dict(target_size=(250, 250), batch_size=16)

valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
valid_generator = valid_datagen.flow_from_directory(
    data_dir, subset="validation", shuffle=False, **dataflow_kwargs)

train_datagen =  tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    **datagen_kwargs
)
train_generator = train_datagen.flow_from_directory(
    data_dir, subset="training", shuffle=False, **dataflow_kwargs)

num_classes = len(train_generator.class_indices)
model = create_model(num_classes)
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=1e-3),
    metrics=['accuracy']
)


early_stopping = EarlyStopping(monitor='val_loss', patience=5)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
tensorboard_callback = create_tensorboard_callback(dir_name="training_logs", experiment_name="emotion_classification")

history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=50,
    callbacks=[early_stopping, lr_scheduler, tensorboard_callback]
)

model.save('my_model')
model.save_weights('my_model_weights.h5')

with zipfile.ZipFile("EfficientNet_test.zip","r") as zip_ref:
    zip_ref.extractall("/EfficientNet_test")

input_folder = 'EfficientNet_test'
target_size = (250, 250)
for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(input_folder, filename)
        img = cv2.imread(input_path)
        img_resized = cv2.resize(img, target_size)
        cv2.imwrite(input_path, img_resized)


claass =['anger','fear','joy','sad','surprise']
for i in range(2,12):
    image = Image.open("EfficientNet_test/{}.jpg".format(i))
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    prediction_scores = model.predict(np.expand_dims(image, axis=0))
    predicted_index = np.argmax(prediction_scores)
    print(prediction_scores)
    print(claass[predicted_index])