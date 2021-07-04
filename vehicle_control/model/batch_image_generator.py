import random

import numpy as np
from matplotlib import image as mpimg

class BatchImageGenerator:

    def __init__(self, image_augmentor):
        self.image_augmentor = image_augmentor

    def batch_generator(self, image_paths, steering_angles, batch_size, is_training):
        """
        The Batch Generator allows us to generate augmented images on the fly, when needed.
        """

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

                im = self.image_augmentor.image_preprocess(im)
                batch_img.append(im)
                batch_steering.append(steering)

            yield (np.asarray(batch_img), np.asarray(batch_steering))
