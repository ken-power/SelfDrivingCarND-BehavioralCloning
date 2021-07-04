from vehicle_control.model.batch_image_generator import BatchImageGenerator
from vehicle_control.model.image_augmentor import ImageAugmentor


class ModelTrainer:

    def __init__(self, model):
        self.model = model

        self.epochs = 10
        self.batch_size = 100
        self.steps_per_epoch = 300  # len(X_train)//batch_size
        self.validation_steps = 200  # len(X_valid)//batch_size

        self.batch_generator = BatchImageGenerator()


    def train_model(self, X_train, y_train, X_valid, y_valid):

        training_generator = self.batch_generator(X_train, y_train, self.batch_size, True)
        validation_generator = self.batch_generator(X_valid, y_valid, self.batch_size, False)

        history = self.model.fit(training_generator,
                                 steps_per_epoch=self.steps_per_epoch,
                                 epochs=self.epochs,
                                 validation_data=validation_generator,
                                 validation_steps=self.validation_steps,
                                 verbose=1,
                                 shuffle=1)

        return history

    def hyperparameters(self):
        return self.epochs, self.batch_size, self.steps_per_epoch, self.validation_steps
