# Behavioral Cloning: Project Writeup


The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image_cnn]: ./images/CNN_architecture.png "CNN Architecture"
[image_model_summary]: ./images/model_summary.png "Model Summary"
[image_model_plot]: ./images/model_plot.png "Model Plot"
[image_model_viz]: ./images/model_viz.png "Model Vizualization"
[image_data_dist]: ./images/data_barchart.png "Data Distribution"
[image_data_dist_normal]: ./images/data_barchart_normalized.png "Normalized Data Distribution"
[image_data_dist_training_val]: ./images/data_barchart_normalized_training_and_validation_sets.png "Distribution of Training and Validation Data Sets"
[image_metrics]: ./images/metrics.png "Metrics"
[image_metrics_latest]: ./images/metrics_latest.png "Latest Metrics"
[image_metrics_lambda_dropout_layers]: ./images/metrics_with_lambda_and_dropout.png "Metrics with Lambda and Dropout Layers"
[image_preprocess]: ./images/preprocessed_image.png "Preprocessed image"
[image_augmented_zoom]: ./images/augment_zoom.png "Image Zooming"
[image_augmented_panning]: ./images/augment_pan.png "Image Panning"
[image_augmented_brightness]: ./images/augment_brightness.png "Image Brightness"
[image_augmented_flipping]: ./images/augment_flip.png "Image Flipping"
[image_augmented_multiple]: ./images/augmented_images_examples.png "Multiple Augmented Images"
[image_center_lane_driving]: data/new/IMG/center_2021_07_02_16_12_37_683.jpg "Center Lane Driving"
[image_left_lane_driving]: data/new/IMG/left_2021_07_02_16_12_37_683.jpg "Left Lane Driving"
[image_right_lane_driving]: data/new/IMG/right_2021_07_02_16_12_37_683.jpg "Right Lane Driving"

---
# Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

## Files Submitted & Code Quality

### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [model.py](vehicle_control/model/model.py) containing the pipeline to create and train the model. The code is split into multiple files that I structureed in a module I called [vehicle_control](vehicle_control). The structure of the [vehicle_control](vehicle_control) module is described below.
* [drive.py](vehicle_control/controller/drive.py) for driving the car in autonomous mode.
* [model.h5](Models/model.h5) containing a trained convolution neural network.
* [writeup.md](writeup.md) summarizing the results (this file that you are reading now).
* [An output video](output_videos/behavioral_cloning_lap_camera_perspective_60fps.mp4) created using [video.py](video.py) that demosntrates the car successfully navigatingthe track. 

The video (uploaded to YouTube) was created from the center camera perspective:

[![Full Lap (60FPS)](https://img.youtube.com/vi/YS8ojkvhBf8/0.jpg)](https://youtu.be/YS8ojkvhBf8 "Video of car driving autonomously for a full lap (60 FPS)")


This animated GIF shows an extract of the car driving autonomously around the track. Videos of the car completing a full lap are below.

![](output_videos/behavioral_cloning_lap.gif)

This video (hosted on YouTube) is sped up 30x and shows the car driving autonomously for a full lap around the track:

[![Full Lap (30x speed)](https://img.youtube.com/vi/6RJoLfOq9N8/0.jpg)](https://youtu.be/6RJoLfOq9N8 "Video of car driving autonomously for a full lap (30x speed)")



### 2. Submission includes functional code

Here is the full structure of the [vehicle_control](vehicle_control) module:
```text
vehicle_control
|
|- controller
    |- drive.py
|
|- model
    |- batch_image_generator.py
    |- data_manager.py
    |- image_augmentor.py
    |- model_builder.py
    |- model_trainer.py
    |- model.py
```

The files in [vehicle_control/model](vehicle_control/model) are:
* [batch_image_generator.py](vehicle_control/model/batch_image_generator.py) contains a batch generator that allows us to generate augmented images on the fly, when needed. 
* [data_manager.py](vehicle_control/model/data_manager.py) is responsible for managing the datasets. It takes in the CVS data file and works with the image files, hiding the details from the rest of the code. The function `training_and_test_data()` provides the training and test data.
* [image_augmentor.py](vehicle_control/model/image_augmentor.py) is responsible for creating the augmented images, [described below in this report](#image-augmentation).
* [model_builder.py](vehicle_control/model/model_builder.py) encapsulates the code that builds the CNN.
* [model_trainer.py](vehicle_control/model/model_trainer.py) is responsible for training the model.
* [model.py](vehicle_control/model/model.py) is the entry point to the vehicle control pipeline, and coordinates the other files in the [vehicle_control/model](vehicle_control/model) module.

I also include a notebook called [behavioral_cloning.ipynb](behavioral_cloning.ipynb) that I created to help with data exploration, testing my code, and analyzing the model results. The notebook also imports and uses the code from the [vehicle_control](vehicle_control) module. 

Using the Udacity-provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python vehicle_control/controller/drive.py Models/model.h5
```

To record images for creating the output video:
```sh
python vehicle_control/controller/drive.py Models/model.h5 output_images
```

Running the program starts a `wsgi` server that communicates with the simulator. **WARNING:** specifying `output_folder` will delete the folder, if already exists, and all of its images, and create a new empty folder. Be sure to back up your images if you want to keep the output images from previous runs.

```text
Creating image folder at output_images
RECORDING THIS RUN ...
(36770) wsgi starting up on http://0.0.0.0:4567
```

When the simulator connects, running `drive.py` from a console displays the steering angle, throttle, and speed for each frame, e.g.:

```text
steering angle: -0.223748 	throttle: 0.076390 	speed: 9.005400
steering angle: -0.243966 	throttle: 0.076950 	speed: 8.999800
steering angle: -0.243966 	throttle: 0.076951 	speed: 8.999800
steering angle: -0.243966 	throttle: 0.076951 	speed: 8.999800
steering angle: -0.192701 	throttle: 0.077217 	speed: 8.997200
steering angle: -0.192701 	throttle: 0.077222 	speed: 8.997200
steering angle: -0.192701 	throttle: 0.077228 	speed: 8.997200
steering angle: 0.022527 	throttle: 0.077488 	speed: 8.994700
steering angle: 0.022527 	throttle: 0.077499 	speed: 8.994700
steering angle: 0.022527 	throttle: 0.077510 	speed: 8.994700
steering angle: 0.196610 	throttle: 0.077194 	speed: 8.997900
steering angle: 0.196610 	throttle: 0.077198 	speed: 8.997900
steering angle: 0.196610 	throttle: 0.077202 	speed: 8.997900
steering angle: 0.160474 	throttle: 0.078155 	speed: 8.988600
steering angle: 0.160474 	throttle: 0.078178 	speed: 8.988600
steering angle: 0.160474 	throttle: 0.078201 	speed: 8.988600
steering angle: 0.112251 	throttle: 0.080182 	speed: 8.969400
steering angle: 0.112251 	throttle: 0.080243 	speed: 8.969400
steering angle: 0.112251 	throttle: 0.080304 	speed: 8.969400
steering angle: 0.025042 	throttle: 0.079937 	speed: 8.973600
steering angle: 0.025042 	throttle: 0.079990 	speed: 8.973600
steering angle: 0.025042 	throttle: 0.080043 	speed: 8.973600
steering angle: -0.024324 	throttle: 0.079055 	speed: 8.983800
steering angle: -0.024324 	throttle: 0.079087 	speed: 8.983800
steering angle: -0.024324 	throttle: 0.079120 	speed: 8.983800
```


### 3. Submission code is usable and readable

The [model.py](vehicle_control/model/model.py) file contains the code for training and saving the convolution neural network. The file contians the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

Here is an extract showing the main pipeline code: 
```python

if __name__ == '__main__':
    print("#### ---- STARTING ---- ####")
    print("#### ---- Retrieving the training data")

    datadir = '../../data/new'
    datafile = 'driving_log.csv'

    data_manager = DataManager(datadir, datafile)
    data_manager.normalize_steering_data()
    X_train, X_valid, y_train, y_valid = data_manager.training_and_test_data()

    print('Training samples: {}'.format(len(X_train)))
    print('Validation samples: {}'.format(len(X_valid)))

    print("#### ---- Building a new model:")

    model_builder = VehicleControlModelBuilder()
    vehicle_control_model = model_builder.nvidia_model()

    print(vehicle_control_model.summary())

    print("#### ---- Training the model:")

    trainer = ModelTrainer(vehicle_control_model)
    trainer.train_model(X_train, y_train, X_valid, y_valid)

    print("#### ---- Saving the trained model:")
    models_dir = 'Models'
    model_name = 'model.h5'

    vehicle_control_model.save(models_dir + '/' + model_name)

    print("#### ---- DONE ---- ####")

```

## Architecture and Training Documentation

### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes `ELU` layers to introduce non-linearity (code line 20). I chose `ELU` rather than `RELU` because research shows `ELU` performs better ([(Clevert, et al., 2015), (Pedamonti, 2018)](#References)). 

I did not use a Keras `Lambda` layer to normalize the data because I already normalized the data before training the model. 

```python
    def nvidia_model(self):
        model = Sequential(name="Vehicle_Control")

        stride_size = (2, 2)

        image_height = 66
        image_width = 200
        number_of_channels = 3
        dropout_rate = 0.5
        learning_rate = 1e-4

        input_dimensions = (image_height, image_width, number_of_channels)

        model.add(Conv2D(24, (5, 5), stride_size,
                         input_shape=input_dimensions,
                         activation='elu', name='Convolutional_feature_map_24_31x98'))

        model.add(Conv2D(36, (5, 5), stride_size, activation='elu', name='Convolutional_feature_map_36_14x47'))
        model.add(Conv2D(48, (5, 5), stride_size, activation='elu', name='Convolutional_feature_map_48_5x22'))
        model.add(Conv2D(64, (3, 3), activation='elu', name='Convolutional_feature_map_64_3x20'))
        model.add(Conv2D(64, (3, 3), activation='elu', name='Convolutional_feature_map_64_1x18'))
        # model.add(Dropout(dropout_rate))

        model.add(Flatten(name='Flatten'))

        model.add(Dense(100, activation='elu', name='Fully_connected_100'))
        # model.add(Dropout(dropout_rate))

        model.add(Dense(50, activation='elu', name='Fully_connected_50'))
        # model.add(Dropout(dropout_rate))

        model.add(Dense(10, activation='elu', name='Fully_connected_10'))
        # model.add(Dropout(dropout_rate))

        # outputs the predicted steering angle for our self-driving car
        model.add(Dense(1, name='Output_vehicle_control'))

        optimizer = Adam(learning_rate=learning_rate)

        model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy', 'mae'])

        return model
```

### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

### 3. Model parameter tuning

The model used an `Adam` optimizer ([model_builder.py](vehicle_control/model/model_builder.py) line 45). I experimented with different learning rates, and eventually settled on a learning rate of `1e-4` ([model_builder.py](vehicle_control/model/model_builder.py) line 17).

```python
optimizer = Adam(learning_rate=learning_rate)

model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy', 'mae'])
```

These are the model hyperparameter values (set in [model_builder.py](vehicle_control/model/model_builder.py)) I ended up using after experimenting with different values:

```text
Model hyperparameters:
Epochs =  5
Batch size = 150
Steps per epoch = 300
Validation Steps = 200
```

### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and changing speeds in different road conditions e.g., slowing on corners, driving faster on long straight stretches.

For details about how I created the training data, see the next section. 

## Model Architecture and Training Strategy

### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with the convolution neural network from the NVIDIA 3-Cmaera model described by [Bojarski, et al (2016)](#References).


In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

The `training_and_test_data()` function in [data_manager.py](vehicle_control/model/data_manager.py) manages this:
```python
    def training_and_test_data(self):
        image_paths, steering_data = self.image_and_steering_data()

        X_train, X_valid, y_train, y_valid = train_test_split(image_paths,
                                                              steering_data,
                                                              test_size=0.2,
                                                              random_state=9)

        return X_train, X_valid, y_train, y_valid

```
This uses the `train_test_split()` function from `sklearn`. This function shuffles the data before splitting. The `random_state` parameter controls the shuffling applied to the data before applying the split. I pass an int (in this case, `9`) so that the output is reproducible across multiple function calls. The `test_size` parameter is a float representing the proportion of the dataset to include in the test split, in this case `0.2` or 20%.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle drove off the track. To improve the driving behavior in these cases, I recorded more training data focusing on these areas.

At the end of the process, the vehicle is now able to drive autonomously around the track without leaving the road.

### 2. Final Model Architecture

The final model architecture ([model_builder.py](vehicle_control/model/model_builder.py) lines 8-49) consisted of a convolution neural network with the layers and layer sizes shown in this summary view of the architecture:

![alt text][image_model_summary]

I created this visualization using `tensorflow.keras.utils.plot_model`:
![alt text][image_model_plot]

I created this visualization using `keras_visuzlizer`: 
![alt text][image_model_viz]

### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving, showing the image captured by the left, center, and right cameras:

Left Camera | Center Camera | Right Camera
:---:|:---:|:---:
![alt text][image_left_lane_driving]|![alt text][image_center_lane_driving]|![alt text][image_right_lane_driving]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover and re-center itself.

Then I repeated this process on track two in order to get more data points.

### Image Augmentation
To augment the data set, I applied several augmentations to the "real" dataset. Applying augmentation techniques is a useful way to create more data from our existing data. This section of the notebook shows how I use zooming, panning, brightness, and flipping to create additional data for training the network. I then randomly apply multiple augmentations to the orginal data, so that, for example, one input image could result in an output image that is a variation that is flipped, rotated, and made brighter.

#### Image Zooming
![alt text][image_augmented_zoom]

#### Image Panning
![alt text][image_augmented_panning]

#### Adjust Image Brightness
![alt text][image_augmented_brightness]

#### Image Flipping
![alt text][image_augmented_flipping]

I wrote the code so that it would randomly apply augmentations, and ensure a reasonable distribution of augmentations. Multiple augmentations can be applied to the same image.

The code for this is in [image_augmentor.py](vehicle_control/model/image_augmentor.py). Here is an extract of the code, showing how I apply the random augmentations. 

```python
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

```

The following shows 10 random examples. The images on the left are the original images captured from the simulator. The images on the right show the results of applying augmentations. The titles show which augmentations have been applied.

![alt text][image_augmented_multiple]



After the collection process, I had over 57,000 data points. I then preprocessed this data by ...

![alt_text][image_data_dist]

As might be expected, this shows that the angle of `0.0`, representing straight ahead, is the most common steering angle. However, for the purposes of training our neural network, this presents a problem because the center value dominate all other values, which would introduce bias in training our network.

The solution is to remove a set of the data from the center of the dataset, which results in a more normalized distribution of the data.

![alt_text][image_data_dist_normal]

The following bar charts show the distribution of the data after splitting the data into training and test sets:
![alt_text][image_data_dist_training_val]

I finally randomly shuffled the data set and put 20% of the data into a validation set.

The code for this is in [data_manager.py](vehicle_control/model/data_manager.py).

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.

## Simulation summary
The car is able to navigate correctly on test data. No tire leaves the drivable portion of the track surface. The car does not pop up onto ledges or roll over any surfaces that would otherwise be considered unsafe.

This is the same video (hosted on YouTube) at normal speed:

[![Full Lap](https://img.youtube.com/vi/TtyE2fUokBQ/0.jpg)](https://youtu.be/TtyE2fUokBQ "Video of car driving autonomously for a full lap")



# References
* Bojarski, M., Del Testa, D., Dworakowski, D., Firner, B., Flepp, B., Goyal, P., Jackel, L.D., Monfort, M., Muller, U., Zhang, J. and Zhang, X., 2016. _End to end learning for self-driving cars_. [arXiv preprint arXiv:1604.07316](https://arxiv.org/pdf/1604.07316.pdf).
* Francois Chollet, 2018. Deep Learning with Python, _Chapter 5: Deep Learning for Computer Vision_. Manning Publications Co. 
* Clevert, D.A., Unterthiner, T. and Hochreiter, S., 2015. _Fast and accurate deep network learning by exponential linear units (elus)_. [arXiv preprint arXiv:1511.07289](https://arxiv.org/pdf/1511.07289.pdf).
* Pedamonti, D., 2018. Comparison of non-linear activation functions for deep neural networks on MNIST classification task. [arXiv preprint arXiv:1804.02763](https://arxiv.org/pdf/1804.02763.pdf).
* Adrian Rosenbrock, 2021. [Visualizing network architectures using Keras and TensorFlow](https://www.pyimagesearch.com/2021/05/22/visualizing-network-architectures-using-keras-and-tensorflow/). pyimagesearch.
