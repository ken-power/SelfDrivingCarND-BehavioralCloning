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
[image_augmented]: ./images/augmented_images_examples.png "Augmented Images"


# Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
## Files Submitted & Code Quality

### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [model.py](vehicle_control/model/model.py) containing the script to create and train the model
* [drive.py](vehicle_control/controller/drive.py) for driving the car in autonomous mode
* [model.h5](Models/model.h5) containing a trained convolution neural network 
* [writeup_report.md](writeup.md) summarizing the results

### 2. Submission includes functional code

Using the Udacity-provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python vehicle_control/controller/drive.py Models/model.h5
```

To record images for creating the output video:
```sh
python vehicle_control/controller/drive.py Models/model.h5 output_images
```

The video (uploaded to YouTube) was created from the center camera perspective:

[![Full Lap (60FPS)](https://img.youtube.com/vi/YS8ojkvhBf8/0.jpg)](https://youtu.be/YS8ojkvhBf8 "Video of car driving autonomously for a full lap (60 FPS)")


This animated GIF shows an extract of the car driving autonomously around the track. Videos of the car completing a full lap are below.

![](output_videos/behavioral_cloning_lap.gif)

This video (hosted on YouTube) is sped up 30x and shows the car driving autonomously for a full lap around the track:

[![Full Lap (30x speed)](https://img.youtube.com/vi/6RJoLfOq9N8/0.jpg)](https://youtu.be/6RJoLfOq9N8 "Video of car driving autonomously for a full lap (30x speed)")

This is the same video (hosted on YouTube) at normal speed:

[![Full Lap](https://img.youtube.com/vi/TtyE2fUokBQ/0.jpg)](https://youtu.be/TtyE2fUokBQ "Video of car driving autonomously for a full lap")



#### 3. Submission code is usable and readable

The [model.py](vehicle_control/model/model.py) file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

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

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an `Adam` optimizer ([model_builder.py](vehicle_control/model/model_builder.py) line 45). I experimented with different learning rates, and eventually settled on a learning rate of `1e-4` ([model_builder.py](vehicle_control/model/model_builder.py) line 17).

```python
optimizer = Adam(learning_rate=learning_rate)

model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy', 'mae'])
```

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a summary of the architecture:

![alt text][image_model]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had over 57,000 data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.





```text
steering angle: -0.211426 	throttle: 0.075952 	speed: 9.010100
steering angle: -0.223748 	throttle: 0.076411 	speed: 9.005400
steering angle: -0.223748 	throttle: 0.076401 	speed: 9.005400
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

# References
* Bojarski, M., Del Testa, D., Dworakowski, D., Firner, B., Flepp, B., Goyal, P., Jackel, L.D., Monfort, M., Muller, U., Zhang, J. and Zhang, X., 2016. _End to end learning for self-driving cars_. [arXiv preprint arXiv:1604.07316](https://arxiv.org/pdf/1604.07316.pdf).
* Francois Chollet, 2018. Deep Learning with Python, _Chapter 5: Deep Learning for Computer Vision_. Manning Publications Co. 
* Clevert, D.A., Unterthiner, T. and Hochreiter, S., 2015. _Fast and accurate deep network learning by exponential linear units (elus)_. [arXiv preprint arXiv:1511.07289](https://arxiv.org/pdf/1511.07289.pdf).
* Pedamonti, D., 2018. Comparison of non-linear activation functions for deep neural networks on MNIST classification task. [arXiv preprint arXiv:1804.02763](https://arxiv.org/pdf/1804.02763.pdf).
* Adrian Rosenbrock, 2021. [Visualizing network architectures using Keras and TensorFlow](https://www.pyimagesearch.com/2021/05/22/visualizing-network-architectures-using-keras-and-tensorflow/). pyimagesearch.
