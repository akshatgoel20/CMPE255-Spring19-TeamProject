import argparse
import pprint
import time

import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from train import train

searchSpace = {
    'decision_function': hp.choice('decision_function', ['ovr', 'ovo']),
    'gamma': hp.uniform('gamma', 0.001, 0.0001),
}

# parse arguments
parser = argparse.ArgumentParser()
# parse arguments
args = parser.parse_args()
# max evaluations
max_evals = int(args.max_evals)
# current arguments
current_eval = 1
# histroiy of training data
train_history = []

# function to get the min of the hyperparams
def function_to_minimize(hyperparams):
    decision_function = hyperparams['decision_function']
    gamma = hyperparams['gamma']
    global current_eval, accuracy, training_time
    global max_evals
    start_time = time.time()
    try:
        accuracy = train()
        training_time = int(round(time.time() - start_time))
        current_eval += 1
        train_history.append(
            {'accuracy': accuracy, 'decision_function': decision_function, 'gamma': gamma, 'time': training_time})
    except Exception as e:
        print("Exception during training: ".format(str(e)))
        np.save("train_history.npy", train_history)
        exit()
    return {'loss': -accuracy, 'time': training_time, 'status': STATUS_OK}

#intialising the trials in trials variable
trials = Trials()
# to get the best trial value
best_trial = fmin(fn=function_to_minimize, space=searchSpace, algo=tpe.suggest, max_evals=max_evals, trials=trials)


#iterating over trials with all the parameters
for trial in trials.trials:
    if trial['misc']['vals']['decision_function'][0] == best_trial['decision_function'] and \
            trial['misc']['vals']['gamma'][0] == best_trial['gamma']:
        best_trial['accuracy'] = -trial['result']['loss'] * 100
        best_trial['time'] = trial['result']['time']
#printing the best trial
pprint.pprint(best_trial)
