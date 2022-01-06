import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.python.keras.layers.preprocessing.image_preprocessing import ResizeMethod


# label_names = ['Abyssinian', 'American Bulldog', 'American Pit Bull Terrier', 'Basset Hound', 'Beagle',
#                'Bengal', 'Birman', 'Bombay', 'Boxer', 'British Shorthair', 'Chihuahua', 'Egyptian Mau',
#                'English Cocker Spaniel', 'English Setter', 'German Shorthaired', 'Great Pyrenees', 'Havanese',
#                'Japanese Chin', 'Keeshond', 'Leonberger', 'Maine Coon', 'Miniature Pinscher', 'New Found Land',
#                'Persian', 'Pomeranian', 'Pug', 'Ragdoll', 'Russian Blue', 'Saint Bernard', 'Samoyed',
#                'Scottish Terrier', 'Shiba Inu', 'Siamese', 'Sphynx', 'Staffordshire Bull Terrier', 'Wheaten Terrier',
#                'Yorkshire Terrier']
label_names = ['Cat', 'Dog']


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, db_dir, batch_size, input_shape, num_classes,
                 shuffle=True, val=False, test=False):
        # you might want to store the parameters into class variables
        self.images_path = db_dir + "/images"
        if test:
            self.labels_path = db_dir + "/annotations/test.txt"
        else:
            self.labels_path = db_dir + "/annotations/trainval.txt"
        self.is_val = val
        self.is_test = test
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        # load the data from the root directory
        self.data, self.labels = self.get_data(db_dir)
        # self.data_val, self.labels_val = self.get_data(db_dir, val=True)
        self.indices = np.array(range(0, len(self.data)))
        self.on_epoch_end()

    def get_data(self, val=False):
        """"
        Loads the paths to the images and their corresponding labels from the database directory
        """
        data = np.array([])
        labels = np.array([])
        with open(self.labels_path) as file:
            for i, row in enumerate(file):
                if (self.is_val and i >= 1843) or (not self.is_val and i < 1843) or self.is_test:
                    row_split = row.split(' ')
                    if len(row_split) == 4 and row_split[2].isdigit():
                        labels = np.append(labels, int(row_split[2]))
                        data = np.append(data, row_split[0])
        self.labels = labels
        self.data = data

        return self.data, self.labels

    def __len__(self):
        """
        Returns the number of batches per epoch: the total size of the dataset divided by the batch size
        """
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        """"
        Generates a batch of data
        """
        batch_x = []
        batch_indices = self.indices[index*self.batch_size : (index+1)*self.batch_size]
        batch_data = []
        for idx in batch_indices:
            batch_data.append(self.data[idx])
        # batch_data = self.data[index*self.batch_size : (index+1)*self.batch_size]
        for file in batch_data:
            image = np.array(Image.open(self.images_path + "/" + file + ".jpg").convert('RGB'))
            if len(image.shape) == 2:
                image = np.squeeze(image)
            image = tf.image.resize_with_pad(image, self.input_shape[0], self.input_shape[1], method=ResizeMethod.BILINEAR, antialias=False)
            image = np.array(array_to_img(image))
            batch_x.append(image)
        batch_x = np.array(batch_x)
        batch_y = self.labels[batch_indices]


        '''
        fig, axes = plt.subplots(nrows=1, ncols=self.batch_size, figsize=[16, 9])
        for i in range(len(axes)):
            axes[i].set_title(label_names[int(batch_y[i]) - 1])
            axes[i].imshow(batch_x[i])
        plt.show()
        '''

        batch_y = tf.one_hot(batch_y, depth=self.num_classes)
        return batch_x, batch_y

    def on_epoch_end(self):
        """"
        Called at the end of each epoch
        """
        # if required, shuffle your data after each epoch
        self.indices = np.arange(len(self.data))
        if self.shuffle:
            # you might find np.random.shuffle useful here
            np.random.shuffle(self.indices)


if __name__ == "__main__":
    train_generator = DataGenerator(db_dir="data", batch_size=4, input_shape=(64,64,3), num_classes=2, shuffle=True)

    batch_x, batch_y = train_generator[0]
    batch_y = tf.argmax(batch_y, axis=1)

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=[16, 9])
    for i in range(len(axes)):
        axes[i].set_title(label_names[int(batch_y[i])-1])
        axes[i].imshow(batch_x[i])
    plt.show()