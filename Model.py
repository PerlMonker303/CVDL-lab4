import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from DataGenerator import DataGenerator
from SGDRScheduler import SGDRScheduler
import tensorflow.python.ops.numpy_ops.np_config as np_config
from utils import evaluateModel, displayGraphs
np_config.enable_numpy_behavior()

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

tf.get_logger().setLevel('INFO')


def resnetBlock(inputs, number_of_filters, filter_dimension, strides):
    x1 = layers.Conv2D(number_of_filters, filter_dimension, padding="same", strides=strides, activation='relu', kernel_initializer="he_normal")(inputs)
    x1 = layers.BatchNormalization()(x1)
    x2 = layers.Conv2D(number_of_filters, filter_dimension, padding="same", strides=strides, activation='relu', kernel_initializer="he_normal")(x1)
    x2 = layers.BatchNormalization()(x2)

    added = layers.Add()([inputs, x2])
    return added


def buildModel(image_shape=(64, 64, 3), no_classes=2):
    img_inputs = keras.Input(shape=image_shape)

    filters = 32
    kernel_size = (3, 3)
    strides = (1, 1)
    x = layers.Conv2D(filters, kernel_size, strides, activation='relu', kernel_initializer="he_normal")(img_inputs)
    x = tf.keras.layers.Dropout(.1)(x)
    x = layers.BatchNormalization()(x)

    x = resnetBlock(x, filters, kernel_size, strides)
    x = layers.MaxPooling2D()(x)
    # x = resnetBlock(x, filters, kernel_size, strides)
    # x = resnetBlock(x, filters, kernel_size, strides)

    x = layers.Conv2D(filters * 2, kernel_size, strides, activation='relu', kernel_initializer="he_normal")(x)
    x = resnetBlock(x, filters * 2, kernel_size, strides)
    x = layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(.1)(x)
    # x = layers.Conv2D(filters * 2, kernel_size, strides, activation='relu')(x)
    # x = tf.keras.layers.Dropout(.1)(x)
    #
    # x = resnetBlock(x, filters * 2, kernel_size, strides)
    # x = resnetBlock(x, filters * 2, kernel_size, strides)
    # x = resnetBlock(x, filters * 2, kernel_size, strides)
    #
    x = layers.Conv2D(filters * 4, kernel_size, strides, activation='relu', kernel_initializer="he_normal")(x)
    x = resnetBlock(x, filters * 4, kernel_size, strides)
    x = tf.keras.layers.Dropout(.1)(x)
    # x = layers.Conv2D(filters * 4, kernel_size, strides, activation='relu')(x)
    # x = tf.keras.layers.Dropout(.1)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(no_classes, activation="relu")(x)
    return tf.keras.Model(img_inputs, x)


acc_history = []
val_acc_history = []
loss_history = []


class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc_history.append(logs.get("acc"))
        val_acc_history.append(logs.get("val_acc"))
        loss_history.append(logs.get("loss"))
        if epoch % 2 == 0:
            print('[Displaying graphs ...]')
            displayGraphs(acc_history, val_acc_history, loss_history)
            print('[... done]')


def train():
    tf.random.set_seed(0)
    batch_size = 128
    input_shape = (32, 32)
    learning_rate = 1e-3
    num_workers = 1
    num_epochs = 200
    save_path = 'saved/'
    load_path = './saved/checkpoint_model_3'
    load_model = False
    label_names = ['Abyssinian', 'American Bulldog', 'American Pit Bull Terrier', 'Basset Hound', 'Beagle',
                   'Bengal', 'Birman', 'Bombay', 'Boxer', 'British Shorthair', 'Chihuahua', 'Egyptian Mau',
                   'English Cocker Spaniel', 'English Setter', 'German Shorthaired', 'Great Pyrenees', 'Havanese',
                   'Japanese Chin', 'Keeshond', 'Leonberger', 'Maine Coon', 'Miniature Pinscher', 'New Found Land',
                   'Persian', 'Pomeranian', 'Pug', 'Ragdoll', 'Russian Blue', 'Saint Bernard', 'Samoyed',
                   'Scottish Terrier', 'Shiba Inu', 'Siamese', 'Sphynx', 'Staffordshire Bull Terrier',
                   'Wheaten Terrier', 'Yorkshire Terrier']
    # label_names = ['Cat', 'Dog']
    train_generator = DataGenerator(db_dir="data", batch_size=batch_size, input_shape=input_shape,
                                    num_classes=len(label_names), shuffle=True)
    val_generator = DataGenerator(db_dir="data", batch_size=batch_size, input_shape=input_shape,
                                    num_classes=len(label_names), shuffle=True, val=True)
    # test_generator = DataGenerator(db_dir="data", batch_size=batch_size, input_shape=input_shape,
    #                                num_classes=len(label_names), shuffle=False, test=True)
    model = buildModel(image_shape=(input_shape[0], input_shape[1], 3), no_classes=len(label_names))
    tf.keras.utils.plot_model(model, to_file="results/model_3.png", show_shapes=True)
    # return

    # lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=learning_rate,
    #     decay_steps=10000,
    #     decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
    model.compile(optimizer, loss='categorical_crossentropy',
                  metrics=['acc'])  # , tf.keras.metrics.TopKCategoricalAccuracy(k=3)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_path + f"checkpoint_model_3", verbose=1)
    if load_model:
        model = tf.keras.models.load_model(load_path)
        # evaluateModel(model, test_generator)
    schedule = SGDRScheduler(min_lr=1e-7, max_lr=1e-3, steps_per_epoch=np.ceil(1843 / batch_size),
                             lr_decay=0.9, cycle_length=15, mult_factor=1.5)
    history = model.fit(x=train_generator, verbose=1, epochs=num_epochs, callbacks=[cp_callback, CustomCallback(), schedule], shuffle=True,
                        batch_size=batch_size, use_multiprocessing=True, workers=num_workers, validation_data=val_generator)

    print('[Displaying graphs ...]')
    displayGraphs(history.history['acc'], history.history['val_acc'], history.history['loss'])
    print('[... done]')


if __name__ == "__main__":
    train()
