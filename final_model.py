from PIL import Image
import os
import tensorflow as tf
import tensorflow_hub as hub
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation,GlobalMaxPooling2D
import numpy as np
from tensorflow.keras.models import load_model
from keras.applications import VGG16
from keras.models import Model
from keras import optimizers , layers, applications
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
import zipfile




with zipfile.ZipFile("datasetFinal.zip","r") as zip_ref:
    zip_ref.extractall("datasetFinal")




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



num_images_Autistic = count_images_in_folder('datasetFinal/Autistic')
num_images_NonAutistic = count_images_in_folder('datasetFinal/Non Autistic')

print("Number of images in the Autistic folder {}\nNumber of images in the Non Autistic folder {}".format(num_images_Autistic,num_images_NonAutistic))

def Data_augmentation(dataset_folder,output_folder,num, parameters):
  if not os.path.exists(output_folder):
      os.makedirs(output_folder)
  datagen = ImageDataGenerator(**parameters)

  file_list = os.listdir(dataset_folder)


  for filename in file_list:
      input_path = os.path.join(dataset_folder, filename)
      img = load_img(input_path)
      img_array = img_to_array(img)

      img_array = img_array.reshape((1,) + img_array.shape)

      augmented_images = datagen.flow(img_array, batch_size=1)

      for i, batch in enumerate(augmented_images):
          augmented_image = array_to_img(batch[0].astype('uint8'))
          output_path = os.path.join(output_folder, f"{filename.split('.')[0]}_aug_{i}.jpg")
          augmented_image.save(output_path)

          if i >= num:
              break


par1= dict(rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

par2 = dict(rotation_range=40,
      zoom_range=0.2)

Data_augmentation('datasetFinal/Non Autistic', 'Dataset/Non Autistic',4,par1)
Data_augmentation('datasetFinal/Autistic', 'Dataset/Autistic',1,par2)

num_images_Autistic = count_images_in_folder('Dataset/Autistic')
num_images_NonAutistic = count_images_in_folder('Dataset/Non Autistic')
print("Number of images in the Autistic folder {}\nNumber of images in the Non Autistic folder {}".format(num_images_Autistic,num_images_NonAutistic))


import tensorflow as tf

data_dir = '/content/Dataset'

datagen_kwargs = dict(rescale=1./255, validation_split=.20)
dataflow_kwargs = dict(target_size=(250, 250), batch_size=32)


valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    **datagen_kwargs
)
new_validation_generator = valid_datagen.flow_from_directory(
    data_dir, subset="validation", shuffle=True, **dataflow_kwargs
)

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    **datagen_kwargs
)

do_data_augmentation = False

if do_data_augmentation:
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=40,
        horizontal_flip=True,
        width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.2, zoom_range=0.2,
        **datagen_kwargs
    )

new_train_generator = train_datagen.flow_from_directory(
    data_dir, subset="training", shuffle=True, **dataflow_kwargs
)



pretrained_model = tf.keras.models.load_model('/content/my_model')
pretrained_model.load_weights('/content/my_model/my_model_weights.h5')


pretrained_model.pop()  
new_output_layer = Dense(2, activation='sigmoid')
pretrained_model.add(new_output_layer)


for layer in pretrained_model.layers[:-1]:
    layer.trainable = False

for layer in pretrained_model.layers:
    layer.trainable = True

pretrained_model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',  
    metrics=['accuracy']
)
 

history_finetune = pretrained_model.fit(
    new_train_generator,
    validation_data=new_validation_generator,
    epochs=10)




