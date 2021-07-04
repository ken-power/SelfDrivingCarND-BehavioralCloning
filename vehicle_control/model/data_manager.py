import ntpath
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class DataManager:

    def __init__(self, data_dir, data_file):
        self.data_dir = data_dir
        self.data_file = data_file

        columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
        pd.set_option('display.max_colwidth', None)

        path_to_datafile = os.path.join(self.data_dir, self.data_file)
        self.data = pd.read_csv(path_to_datafile, names=columns)

        # Remove the path from the filename entries of the images, leaving just the filename
        self.data['center'] = self.data['center'].apply(self.path_leaf)
        self.data['left'] = self.data['left'].apply(self.path_leaf)
        self.data['right'] = self.data['right'].apply(self.path_leaf)

    def get_data(self):
        return self.data

    def steering_data(self):
        return self.data['steering']

    def throttle_data(self):
        return self.data['throttle']

    def reverse_data(self):
        return self.data['reverse']

    def speed_data(self):
        return self.data['speed']

    def path_leaf(self, path):
        head, tail = ntpath.split(path)
        return tail

    def normalize_steering_data(self):
        print('total data', len(self.data))
        remove_list = []

        num_bins = 25
        max_samples_per_bin = 400
        hist, bins = np.histogram(self.data['steering'], num_bins)

        for j in range(num_bins):
            list_ = []

            for i in range(len(self.data['steering'])):
                if self.data['steering'][i] >= bins[j] and self.data['steering'][i] <= bins[j + 1]:
                    list_.append(i)

            list_ = shuffle(list_)
            list_ = list_[max_samples_per_bin:]
            remove_list.extend(list_)

        print('removed:', len(remove_list))

        self.data.drop(self.data.index[remove_list], inplace=True)

        print('remaining:', len(self.data))

    def training_and_test_data(self):
        image_paths, steering_data = self.image_and_steering_data()

        X_train, X_valid, y_train, y_valid = train_test_split(image_paths,
                                                              steering_data,
                                                              test_size=0.2,
                                                              random_state=9)

        return X_train, X_valid, y_train, y_valid

    def image_and_steering_data(self):
        image_path = []
        steering = []

        for i in range(len(self.data)):
            indexed_data = self.data.iloc[i]
            center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]

            image_path.append(os.path.join(self.data_dir + '/IMG/', center.strip()))
            steering.append(float(indexed_data[3]))

        image_paths = np.asarray(image_path)
        steering_data = np.asarray(steering)

        return image_paths, steering_data
