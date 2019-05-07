import argparse
import os
import time
import _pickle as cPickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from parameters import DATASET, TRAINING, HYPERPARAMS
import numpy as np


def load_data(validation=False, test=False):
    data_dict = dict()
    validation_dict = dict()
    test_dict = dict()
    features = "landmarks_and_hog"

    if DATASET.name == "Fer2013":
        data_dict['X'] = np.load(DATASET.train_folder + '/landmarks.npy')
        data_dict['X'] = np.array([x.flatten() for x in data_dict['X']])
        data_dict['X'] = np.concatenate((data_dict['X'], np.load(DATASET.train_folder + '/hog_features.npy')),
                                            axis=1)

        data_dict['X'] = np.load(DATASET.train_folder + '/landmarks.npy')
        data_dict['X'] = np.array([x.flatten() for x in data_dict['X']])

        data_dict['X'] = np.load(DATASET.train_folder + '/hog_features.npy')
        data_dict['Y'] = np.load(DATASET.train_folder + '/labels.npy')

        if DATASET.trunc_trainset_to > 0:
            data_dict['X'] = data_dict['X'][0:DATASET.trunc_trainset_to, :]
            data_dict['Y'] = data_dict['Y'][0:DATASET.trunc_trainset_to]
        if validation:
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
                print("Error '{}' features not recognized".format(HYPERPARAMS.features))
            validation_dict['Y'] = np.load(DATASET.validation_folder + '/labels.npy')
            if DATASET.trunc_validationset_to > 0:
                validation_dict['X'] = validation_dict['X'][0:DATASET.trunc_validationset_to, :]
                validation_dict['Y'] = validation_dict['Y'][0:DATASET.trunc_validationset_to]
        if test:
            # load train set
            if features == "landmarks_and_hog":
                test_dict['X'] = np.load(DATASET.test_folder + '/landmarks.npy')
                test_dict['X'] = np.array([x.flatten() for x in test_dict['X']])
                test_dict['X'] = np.concatenate((test_dict['X'], np.load(DATASET.test_folder + '/hog_features.npy')),
                                                axis=1)
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


def train(train_model=True):
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
        validation_accuracy = evaluate(model, validation['X'], validation['Y'])
        print("  - validation accuracy = {0:.1f}".format(validation_accuracy * 100))
        return validation_accuracy
    else:
        # Testing phase : load saved model and evaluate on test dataset
        print("start evaluation...")
        print("loading pretrained model...")
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
        start_time = time.time()
        validation_accuracy = evaluate(model, validation['X'], validation['Y'])
        print("  - validation accuracy = {0:.1f}".format(validation_accuracy * 100))
        test_accuracy = evaluate(model, test['X'], test['Y'])
        print("  - test accuracy = {0:.1f}".format(test_accuracy * 100))
        print("  - evalution time = {0:.1f} sec".format(time.time() - start_time))
        return test_accuracy


def evaluate(model, X, Y):
    predicted_Y = model.predict(X)
    accuracy = accuracy_score(Y, predicted_Y)
    return accuracy


# parse arg to see if we need to launch training now or not yet
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", default="no")
parser.add_argument("-e", "--evaluate", default="no")
parser.add_argument("-m", "--max_evals")
args = parser.parse_args()
if args.train == "yes" or args.train == "Yes" or args.train == "YES":
    train()
if args.evaluate == "yes" or args.evaluate == "Yes" or args.evaluate == "YES":
    train(train_model=False)
