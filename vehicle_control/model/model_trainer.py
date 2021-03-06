from vehicle_control.model.batch_image_generator import BatchImageGenerator
from vehicle_control.model.image_augmentor import ImageAugmentor

from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger

class ModelTrainer:

    def __init__(self, model):
        self.model = model

        self.epochs = 5
        self.batch_size = 256
        self.steps_per_epoch = 300
        self.validation_steps = 200

        image_augmentor = ImageAugmentor()
        self.image_generator = BatchImageGenerator(image_augmentor)


    def train_model(self, X_train, y_train, X_valid, y_valid):

        training_generator = self.image_generator.batch_generator(X_train, y_train, self.batch_size, True)
        validation_generator = self.image_generator.batch_generator(X_valid, y_valid, self.batch_size, False)

        # define callbacks to save history and weights
        checkpoint_recorder = ModelCheckpoint('../../checkpoints/weights.{epoch:02d}-{val_loss:.3f}.hdf5')
        logger = CSVLogger(filename='../../logs/history.csv')

        vehicle_control_callbacks = [checkpoint_recorder, logger]

        history = self.model.fit(training_generator,
                                 steps_per_epoch=self.steps_per_epoch,
                                 epochs=self.epochs,
                                 validation_data=validation_generator,
                                 validation_steps=self.validation_steps,
                                 verbose=1,
                                 shuffle=1,
                                 callbacks=vehicle_control_callbacks)

        return history

    def hyperparameters(self):
        return self.epochs, self.batch_size, self.steps_per_epoch, self.validation_steps
