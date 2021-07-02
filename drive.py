import argparse
import base64
import os
import shutil
from datetime import datetime
from io import BytesIO

import cv2
import eventlet.wsgi
import h5py
import numpy as np
import socketio
from PIL import Image
from flask import Flask
from keras import __version__ as keras_version
from keras.models import load_model

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

speed_limit = 15

class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral


controller = SimplePIController(0.1, 0.002)
set_speed = 9
controller.set_desired(set_speed)


def image_preprocess(image):
    top_of_image = 60
    bottom_of_image = 135
    image = image[top_of_image:bottom_of_image, :, :]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

    kernel_size = (3, 3)
    image = cv2.GaussianBlur(image, kernel_size, 0)

    target_size = (200, 66)  # per NVidia model recommendations
    image = cv2.resize(image, target_size)

    # normalize the image
    image = image / 255

    return image


def save_image_to_file(image, filename):
    print("***FILENAME={}, image shape={}".format(filename, image.shape))

    im = Image.fromarray(image[1:])
    im.save(filename)


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]

        # The current throttle of the car
        throttle = data["throttle"]

        # The current speed of the car
        speed = float(data["speed"])

        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image = np.asarray(image)
        image = image_preprocess(image)
        image = np.array([image])  # expects a 4D array, hence wrap image in np.array

#        steering_angle = float(model.predict(image[None, :, :, :], batch_size=1))
#        throttle = controller.update(float(speed))
        steering_angle = float(model.predict(image))
        throttle = 1.0 - speed / speed_limit

        print('steering angle: {:.6f} \tthrottle: {:.6f} \tspeed: {:.6f}'.format(steering_angle, throttle, speed))

        send_control(steering_angle, throttle)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image_filename = image_filename + '.jpg'
            save_image_to_file(image, image_filename)
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    # check that model Keras version is same as local Keras version
    f = h5py.File(args.model, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)

    model = load_model(args.model)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
