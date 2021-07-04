import cv2
import numpy as np
from imgaug import augmenters as ima
from matplotlib import image as mpimg


class ImageAugmentor:

    def __init__(self):
        self.top_of_image = 60
        self.bottom_of_image = 135

    def image_preprocess(self, image):
        image = image[self.top_of_image:self.bottom_of_image, :, :]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

        kernel_size = (3, 3)
        image = cv2.GaussianBlur(image, kernel_size, 0)

        target_size = (200, 66)  # per NVidia model recommendations
        image = cv2.resize(image, target_size)

        # normalize the image
        image = image / 255

        return image

    def zoom(self, image):
        zoom = ima.Affine(scale=(1, 1.3))
        image = zoom.augment_image(image)
        return image

    def pan(self, image):
        pan = ima.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
        image = pan.augment_image(image)
        return image

    def randomly_alter_brightness(self, image):
        brightness = ima.Multiply((0.2, 1.2))
        image = brightness.augment_image(image)
        return image

    def flip(self, image, steering_angle):
        HORIZONTAL_FLIP = 1
        image = cv2.flip(image, HORIZONTAL_FLIP)
        steering_angle = -steering_angle
        return image, steering_angle

    def random_augment(self, image, steering_angle):
        augmentation_types = []

        # we don't want any one transform used more than 50% of the time
        FREQUENCY_THRESHOLD = 0.5

        image = mpimg.imread(image)

        if np.random.rand() < FREQUENCY_THRESHOLD:
            image = self.pan(image)
            augmentation_types.append("Pan")
        if np.random.rand() < FREQUENCY_THRESHOLD:
            image = self.zoom(image)
            augmentation_types.append("Zoom")
        if np.random.rand() < FREQUENCY_THRESHOLD:
            image = self.randomly_alter_brightness(image)
            augmentation_types.append("Brightness")
        if np.random.rand() < FREQUENCY_THRESHOLD:
            image, steering_angle = self.flip(image, steering_angle)
            augmentation_types.append("Flip")

        return image, steering_angle, augmentation_types
