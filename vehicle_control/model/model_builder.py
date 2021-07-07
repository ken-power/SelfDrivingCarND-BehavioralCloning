from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense, Lambda, Dropout, MaxPool2D
from tensorflow.python.keras.optimizer_v2.adam import Adam


class VehicleControlModelBuilder:

    def nvidia_model(self):

        stride_size = (2, 2)

        image_height = 66
        image_width = 200
        number_of_channels = 3
        learning_rate = 1e-4

        input_dimensions = (image_height, image_width, number_of_channels)

        model = Sequential(name="Vehicle_Control")

        # model.add(Lambda(lambda x: x/127.5 -1., input_shape=input_dimensions))

        model.add(Conv2D(24, (5, 5), stride_size, input_shape=input_dimensions, activation='elu', name='Convolutional_feature_map_24_31x98'))
        model.add(Conv2D(36, (5, 5), stride_size, activation='elu', name='Convolutional_feature_map_36_14x47'))
        model.add(Conv2D(48, (5, 5), stride_size, activation='elu', name='Convolutional_feature_map_48_5x22'))
        model.add(Conv2D(64, (3, 3), activation='elu', name='Convolutional_feature_map_64_3x20'))
        model.add(Conv2D(64, (3, 3), activation='elu', name='Convolutional_feature_map_64_1x18'))
        # model.add(Dropout(dropout_rate))
        # model.add(MaxPool2D(name='Max_pooling'))

        model.add(Flatten(name='Flatten'))

        model.add(Dense(100, activation='elu', name='Fully_connected_100'))
        # model.add(Dropout(0.2))

        model.add(Dense(50, activation='elu', name='Fully_connected_50'))
        # model.add(Dropout(dropout_rate))

        model.add(Dense(10, activation='elu', name='Fully_connected_10'))
        model.add(Dropout(0.2))

        # outputs the predicted steering angle for our self-driving car
        model.add(Dense(1, name='Output_vehicle_control'))

        optimizer = Adam(learning_rate=learning_rate)

        model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy', 'mae'])

        return model

