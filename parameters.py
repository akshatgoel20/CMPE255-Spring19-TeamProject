# initiliasing the collection class with all the parameters
class Collection:
    name = 'Fer2013'
    train_folder = 'fer2013_features/Training'
    validation_folder = 'fer2013_features/PublicTest'
    test_folder = 'fer2013_features/PrivateTest'
    trunc_trainset_to = -1
    trunc_validationset_to = -1
    trunc_testset_to = -1


# initiliasing the Hyperparameters class with all the parameters
class Hyperparameters:
    random_state = 0
    epochs = 15000
    epochs_during_hyperopt = 600
    kernel = 'rbf'
    decision_function = 'ovr'
    features = "landmarks_and_hog"
    gamma = 'auto'

# initiliasing the train class with all the parameters
class Train:
    save_model = True
    save_model_path = "saved_model.bin"

#for the DATASET
DATASET = Collection()
#for the TRAINING
TRAINING = Train()
#for the HYPERPARAMS
HYPERPARAMS = Hyperparameters()
