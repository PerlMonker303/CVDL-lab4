import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score

from DataGenerator import DataGenerator


def ensemble_predictions(members, testX):
    # make predictions
    yhats = [model.predict(testX) for model in members]
    yhats = np.array(yhats)
    # sum across ensemble members
    summed = np.sum(yhats, axis=0)
    # argmax across classes
    result = np.argmax(summed, axis=1)
    return result


def ensembleValidation():
    batch_size = 16
    input_shape = (32, 32)
    model_paths = ['./saved/checkpoint_model_2_ok', './saved/checkpoint_model_3_ok',
                   './saved/checkpoint_model_4_ok', './saved/checkpoint_model_10_ok']

    # load models
    models = []
    for path in model_paths:
        model = tf.keras.models.load_model(path)
        models.append(model)

    # create validation data loader
    val_generator = DataGenerator(db_dir="data", batch_size=batch_size, input_shape=input_shape,
                                  num_classes=37, shuffle=True, val=True)

    accuracy_total = 0
    length = len(val_generator)
    for x, y in val_generator:
        yhat = ensemble_predictions(models, x)
        # yht = tf.one_hot(yhat, depth=37)
        y = tf.argmax(y, axis=1)
        acc = accuracy_score(y, yhat)
        accuracy_total += acc

    accuracy_total /= length
    print(f"Accuracy: {np.mean(accuracy_total)}")


if __name__ == "__main__":
    ensembleValidation()