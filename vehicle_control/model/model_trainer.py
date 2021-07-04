import random

import numpy as np
from matplotlib import image as mpimg

from vehicle_control.model.image_augmentor import ImageAugmentor


class ModelTrainer:

    def __init__(self, model):
        self.model = model

        self.steps_per_epoch = 300  # len(X_train)//batch_size
        self.validation_steps = 200  # len(X_valid)//batch_size
        self.epochs = 10
        self.image_augmentor = ImageAugmentor()

    def batch_generator(self, image_paths, steering_angles, batch_size, is_training):

        while True:
            batch_img = []
            batch_steering = []

            for i in range(batch_size):
                random_index = random.randint(0, len(image_paths) - 1)

                if is_training:
                    im, steering, aug_type = \
                        self.image_augmentor.random_augment(image_paths[random_index],
                                            steering_angles[random_index])
                else:
                    im = mpimg.imread(image_paths[random_index])
                    steering = steering_angles[random_index]

                im = self.image_preprocess(im)
                batch_img.append(im)
                batch_steering.append(steering)

            yield (np.asarray(batch_img), np.asarray(batch_steering))

    def train_model(self, X_train, y_train, X_valid, y_valid):
        batch_size = 100

        training_generator = self.batch_generator(X_train, y_train, batch_size, True)
        validation_generator = self.batch_generator(X_valid, y_valid, batch_size, False)

        history = self.model.fit(training_generator,
                                 steps_per_epoch=self.steps_per_epoch,
                                 epochs=self.epochs,
                                 validation_data=validation_generator,
                                 validation_steps=self.validation_steps,
                                 verbose=1,
                                 shuffle=1)

        return history
