import tensorflow as tf
from tensorflow import keras 

def load_dataset(dataset_main_dir, batch_size=32, image_size=(60,60),
                validation_split=0.2):
  """
    Load training and validation set each as Tensorflow Dataset 
    with labels taken from folders names.

    Arguments:
      -- dataset_main_dir - main directory of the dataset directory structure
      -- batch_size - size of one batch of images to load
      -- image_size - size to resize images to after read from the disk

    Returns:
      -- (train_ds, validation_ds) - Python tuple with training and validation
          Tensorflow based datasets.
  """

  train_ds = tf.keras.utils.image_dataset_from_directory(
    directory=dataset_main_dir,
    label_mode='categorical',
    batch_size=batch_size,
    image_size=image_size,
    validation_split=validation_split,
    subset='training',
    seed=44
  )

  validation_ds = tf.keras.utils.image_dataset_from_directory(
    directory=dataset_main_dir,
    label_mode='categorical',
    batch_size=batch_size,
    image_size=image_size,
    validation_split=validation_split,
    subset='validation',
    seed=44
  )

  return (train_ds, validation_ds)

def preprocess_image(image):
  """
    Image preprocessing. Each pixel value inside an image is processed in such
    a way that it goes into interval [0, 1]. 

    Arguments:
      -- image - Input image

    Returns:
      -- preprocessed_img - Preprocessed image with pixel values set 
        appropriately
  """

  preprocessed_img = tf.image.convert_image_dtype(image, tf.float32) / 255.

  return preprocessed_img

  


