import _pickle as cPickle
import argparse
import os
import time

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from parameters import DATASET, TRAINING, HYPERPARAMS


# loading the data with validation as false and test as also as false
def load_data(validation=False, test=False):
    # getting the dict
    data_dict = dict()
    # getting the validation dict
    validation_dict = dict()
    # getting the test dict
    test_dict = dict()
    # getting the features
    features = "landmarks_and_hog"

    # loading the dataset
    if DATASET.name == "Fer2013":
        # loading the dataset for landmarks
        data_dict['X'] = np.load(DATASET.train_folder + '/landmarks.npy')
        # loading the dataset for array
        data_dict['X'] = np.array([x.flatten() for x in data_dict['X']])
        # loading the dataset for hog features
        data_dict['X'] = np.concatenate((data_dict['X'], np.load(DATASET.train_folder + '/hog_features.npy')),
                                        axis=1)
        data_dict['X'] = np.load(DATASET.train_folder + '/hog_features.npy')
        # loading the dataset for labels
        data_dict['Y'] = np.load(DATASET.train_folder + '/labels.npy')
        # loading the dataset for landmarks
        data_dict['X'] = np.load(DATASET.train_folder + '/landmarks.npy')
        data_dict['X'] = np.array([x.flatten() for x in data_dict['X']])

        # if the there is a data in the dataset then defingin data_dict
        if DATASET.trunc_trainset_to > 0:
            data_dict['X'] = data_dict['X'][0:DATASET.trunc_trainset_to, :]
            data_dict['Y'] = data_dict['Y'][0:DATASET.trunc_trainset_to]
        if validation:
            # after validation initialising features to landmarks and hog
            if features == "landmarks_and_hog":
                validation_dict['X'] = np.load(DATASET.validation_folder + '/landmarks.npy')
                validation_dict['X'] = np.array([x.flatten() for x in validation_dict['X']])
                validation_dict['X'] = np.concatenate(
                    (validation_dict['X'], np.load(DATASET.validation_folder + '/hog_features.npy')), axis=1)
            elif features == "landmarks":
                validation_dict['X'] = np.load(DATASET.validation_folder + '/landmarks.npy')
                validation_dict['X'] = np.array([x.flatten() for x in validation_dict['X']])
            elif features == "hog":
                validation_dict['X'] = np.load(DATASET.validation_folder + '/hog_features.npy')
            else:
                print("Features not recognized".format(HYPERPARAMS.features))
            validation_dict['Y'] = np.load(DATASET.validation_folder + '/labels.npy')
            if DATASET.trunc_validationset_to > 0:
                validation_dict['X'] = validation_dict['X'][0:DATASET.trunc_validationset_to, :]
                validation_dict['Y'] = validation_dict['Y'][0:DATASET.trunc_validationset_to]
        if test:
            # load landmarks and hog train  set
            if features == "landmarks_and_hog":
                # load landmarks train  set
                test_dict['X'] = np.load(DATASET.test_folder + '/landmarks.npy')
                test_dict['X'] = np.array([x.flatten() for x in test_dict['X']])
                test_dict['X'] = np.concatenate((test_dict['X'], np.load(DATASET.test_folder + '/hog_features.npy')),
                                                axis=1)
                # load hyperparameters for landmarks and hog set
            elif HYPERPARAMS.features == "landmarks":
                test_dict['X'] = np.load(DATASET.test_folder + '/landmarks.npy')
                test_dict['X'] = np.array([x.flatten() for x in test_dict['X']])
            elif HYPERPARAMS.features == "hog":
                test_dict['X'] = np.load(DATASET.test_folder + '/hog_features.npy')
            else:
                print("Error '{}' features not identified".format(HYPERPARAMS.features))
            test_dict['Y'] = np.load(DATASET.test_folder + '/labels.npy')
            np.save(DATASET.test_folder + "/lab.npy", test_dict['Y'])
            if DATASET.trunc_testset_to > 0:
                test_dict['X'] = test_dict['X'][0:DATASET.trunc_testset_to, :]
                test_dict['Y'] = test_dict['Y'][0:DATASET.trunc_testset_to]

        if not validation and not test:
            return data_dict
        elif not test:
            return data_dict, validation_dict
        else:
            return data_dict, validation_dict, test_dict
    else:
        print("Dataset is not found")
        exit()


# function to train the dataset
def train(train_model=True):
    global model, test
    #for 9000 iterations
    epochs = 9000
    print("loading dataset: Fer2013")
    if train_model:
        data, validation = load_data(validation=True)
    else:
        data, validation, test = load_data(validation=True, test=True)

    if train_model:
        print("building model")
        model = SVC(random_state=1234, max_iter=epochs, kernel='rbf',
                    decision_function_shape='ovr', gamma='auto')

        print("start training...")
        print("--")
        print("max epochs: {} ".format(epochs))
        print("--")
        print("Training samples: {}".format(len(data['Y'])))
        print("Validation samples: {}".format(len(validation['Y'])))
        print("--")
        start_time = time.time()
        model.fit(data['X'], data['Y'])
        training_time = time.time() - start_time
        print("training time = {0:.1f} sec".format(training_time))

        if TRAINING.save_model:
            print("saving model...")
            with open(TRAINING.save_model_path, 'wb') as f:
                cPickle.dump(model, f)

        print("evaluating...")
        predicted_y = model.predict(validation['X'])
        validation_accuracy = accuracy_score(validation['Y'], predicted_y)

        print("  - validation accuracy = {0:.1f}".format(validation_accuracy * 100))
        return validation_accuracy
    else:
        # Testing phase : load saved model and evaluate on test dataset
        print("start evaluation...")
        print("loading pre-trained model...")
        if os.path.isfile(TRAINING.save_model_path):
            with open(TRAINING.save_model_path, 'rb') as f:
                model = cPickle.load(f)
        else:
            print("Error: file '{}' not found".format(TRAINING.save_model_path))
            exit()

        print("--")
        print("Validation samples: {}".format(len(validation['Y'])))
        print("Test samples: {}".format(len(test['Y'])))
        print("--")
        print("evaluating...")
        #getting the start time
        start_time = time.time()
        predicted_y = model.predict(validation['X'])
        #vallidation accruacy
        validation_accuracy = accuracy_score(validation['Y'], predicted_y)
        print("  - validation accuracy = {0:.1f}".format(validation_accuracy * 100))
        predicted_y = model.predict(test['Y'])
        #testing accuracy
        test_accuracy = accuracy_score(validation['Y'], predicted_y)
        print("  - test accuracy = {0:.1f}".format(test_accuracy * 100))
        print("  - evaluation time = {0:.1f} sec".format(time.time() - start_time))
        return test_accuracy


# parse the argumenst for train evaluae and maximum evaluations needed
parser = argparse.ArgumentParser()
#train arguments
parser.add_argument("-t", "--train", default="no")
#evaluare arguments
parser.add_argument("-e", "--evaluate", default="no")
# evals arguments
parser.add_argument("-m", "--max_evals")
args = parser.parse_args()
if args.train == "yes" or args.train == "Yes" or args.train == "YES":
    train()
if args.evaluate == "yes" or args.evaluate == "Yes" or args.evaluate == "YES":
    train(train_model=False)
