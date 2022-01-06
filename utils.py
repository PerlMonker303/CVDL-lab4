import matplotlib.pyplot as plt
from tensorflow import keras


def evaluateModel(model, test_generator):
    evaluation = model.evaluate(x=test_generator, return_dict=True, batch_size=16, verbose=1)
    for name, value in evaluation.items():
        print(f"{name}: {value:.4f}")


def displayGraphs(acc, val_acc, loss):
    # Plot model accuracy
    plt.plot(acc)
    plt.plot(val_acc)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # Plot loss
    plt.plot(loss)
    # plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()
