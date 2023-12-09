import numpy as np
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.layers import Dense
from sklearn.model_selection import KFold

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
    label_mode='int',
    batch_size=batch_size,
    image_size=image_size,
    shuffle=False,
    validation_split=validation_split,
    subset='training',
    seed=42
  )

  validation_ds = tf.keras.utils.image_dataset_from_directory(
    directory=dataset_main_dir,
    label_mode='int',
    batch_size=batch_size,
    image_size=image_size,
    shuffle=False,
    validation_split=validation_split,
    subset='validation',
    seed=42
  )

  return (train_ds, validation_ds)

def preprocess_image(image, label):
  """
    Image preprocessing. Each pixel value inside an image is processed in such
    a way that it goes into interval [0, 1]. 

    Arguments:
      -- image - Input image
      -- label - Label assigned to an image from dataset

    Returns:
      -- preprocessed_img - Preprocessed image with pixel values set 
        appropriately
  """

  preprocessed_img = tf.image.convert_image_dtype(image, tf.float32) / 255.

  return preprocessed_img, label

def get_mlp_layers(model):
  """
    Get layers of a Multi-Layer Perceptron from a model.

    Arguments:
      -- model - Tensorflow's sequential neural network model
      
    Returns:
      -- mlp_layers - Array with Multi-Layer Perceptron layers taken from the
                        model
  """

  mlp_start_idx = 0
  for i, layer in enumerate(model.layers):
    if isinstance(layer, Dense):
      mlp_start_idx = i - 1 # with Flatten layer
      break
    
  mlp_layers = model.layers[mlp_start_idx:]
  return mlp_layers

def kfold(inputs, targets, Model, num_folds=5, num_epochs=1500,
          early_stopping=False):
  """
    Performs k-fold cross validation to properly assess model performance.

    Arguments:
      -- inputs - dataset on which k-fold will be executed
      -- targets - labels for inputs
      -- Model - reference to a model's class. This model will be validated
      -- num_folds - number of folds 
      -- num_epochs - number of epochs for training a model
      -- early_stopping - True if early stopping is enabled to prevent overfitting

    Returns:
      -- score - average model's accuracy
      -- mean_loss - average model's loss
  """
  k_fold = KFold(n_splits=num_folds, shuffle=True)

  accuracy_scores = []
  losses = []

  for fold, (train, val) in enumerate(k_fold.split(inputs, targets)):
    print(f"Fold {fold + 1}/{num_folds}")

    model = Model()

    # Wyzerowanie wag modelu
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    #print('Training shape: ', inputs[train].shape)
    #print('Validation shape: ', inputs[val].shape)

    if early_stopping:
      model.fit(inputs[train],
                  targets[train],
                  batch_size=80,
                  epochs=num_epochs,
                  validation_data=(inputs[val], targets[val]),
                  verbose=0,
                  callbacks=[model.early_stopping])
    else:
      model.fit(inputs[train],
                  targets[train],
                  batch_size=80,
                  epochs=num_epochs,
                  validation_data=(inputs[val], targets[val]),
                  verbose=0)

    loss, accuracy_score = model.evaluate(inputs[val], targets[val])
    accuracy_scores.append(accuracy_score)
    losses.append(loss)
    
  score = np.mean(accuracy_scores)
  mean_loss = np.mean(losses)

  return score, mean_loss

def convert_tf_ds_to_numpy(tf_ds):
  """
    Converts Tensorflow's dataset to two numpy arrays - one for samples, one for
    labels.

    Arguments:
      -- tf_ds - dataset in a Tensorflow format

    Returns:
      -- samples - samples as numpy array
      -- labels - labels as numpy array
  """
  samples = []
  labels = []

  # Pobranie każdego elementu z tfds w postaci tablicy numpy
  for images, tf_labels in tf.data.Dataset.as_numpy_iterator(tf_ds):
    # Rozłożenie każdego batcha na pojedyncze obrazy
    samples.extend(images)
    labels.extend(tf_labels)

  # Zapisanie obrazów 3D jako wektory 1D
  samples = np.array([img.flatten() for img in samples])
  labels = np.array(labels)

  return (samples, labels)

  


