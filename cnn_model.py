class CNNClasifier(tf.keras.Model):
  def __init__(self, num_filters=30, dense_units=20, num_classes=3):
    super().__init__()

    self.preprocessing = Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.RandomFlip(),
        tf.keras.layers.RandomRotation(0.3),
        tf.keras.layers.RandomZoom(0.3)
    ])

    self.conv1 = tf.keras.layers.Conv2D(num_filters, # number of filters (30)
                                        3, # kernel size
                                        activation='relu',
                                        kernel_initializer="he_normal")

    self.max_pool1 = tf.keras.layers.MaxPooling2D(strides=2)

    self.conv2 = tf.keras.layers.Conv2D(num_filters*2, # number of filters (60)
                                        3, # kernel size
                                        activation='relu',
                                        kernel_initializer="he_normal")

    self.dropout1 = tf.keras.layers.Dropout(0.2)

    self.max_pool2 = tf.keras.layers.MaxPooling2D(strides=2)

    self.conv3 = tf.keras.layers.Conv2D(num_filters*3, # number of filters (90)
                                        3, # kernel size
                                        strides=(2,2),
                                        activation='relu',
                                        kernel_initializer="he_normal")

    self.dropout2 = tf.keras.layers.Dropout(0.4)

    self.flatten = tf.keras.layers.Flatten()

    self.dense1 = tf.keras.layers.Dense(dense_units,
                                       activation='relu',
                                       kernel_initializer="he_normal")

    self.dropout3 = tf.keras.layers.Dropout(0.3)

    self.dense2 = tf.keras.layers.Dense(dense_units*0.5,
                                        activation='relu',
                                        kernel_initializer="he_normal")

    self.output_layer = tf.keras.layers.Dense(num_classes,
                                              activation='linear')

  # Forward pass
  def call(self, inputs, training=False):
    x = self.preprocessing(inputs)
    x = self.conv1(x)
    x = self.max_pool1(x)
    x = self.conv2(x)
    if training:
     x = self.dropout1(x)
    x = self.max_pool2(x)
    x = self.conv3(x)
    if training:
      x = self.dropout2(x)
    x = self.flatten(x)
    x = self.dense1(x)
    if training:
      x = self.dropout3(x)
    x = self.dense2(x)
    return self.output_layer(x)
