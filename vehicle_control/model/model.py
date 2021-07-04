from sklearn.model_selection import train_test_split

from vehicle_control.model.data_manager import DataManager
from vehicle_control.model.model_builder import VehicleControlModelBuilder
from vehicle_control.model.model_trainer import ModelTrainer

if __name__ == '__main__':

    print("#### ---- Retrieving the training data")

    datadir = 'data/new'
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
