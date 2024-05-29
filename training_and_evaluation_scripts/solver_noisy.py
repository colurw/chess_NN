""" Adds random noise to testing data, in order to produce multiple average-able solves from 
one neural network.  Somewhat effective, but bested by an ensemble of solvers. """

import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
import random
from _0_chess_tools import one_hot_to_unicode

# Get data from chess_encoder.py
print('loading pickled data...')
with open("mate_in_1.pickle", "rb") as file:
    mate_in_one = pickle.load(file) 
with open("solution.pickle", "rb") as file:
    solution = pickle.load(file) 

# Create datasets
print('\ncreating test dataset...')
x_test = np.array(mate_in_one[1300000:], dtype='float')
y_test = np.array(solution[1300000:], dtype='float')
print(x_test.shape, y_test.shape, '\n')

# Load model from chess_trainer.py
model = keras.models.load_model('models/matein1')
# Calculate ensemble accuracy
ensemble = 100
count = 0
for i in (range(500)):
    # Get random board
    rand_num = random.sample(range(len(x_test)), 1)
    x_sample = x_test[rand_num]
    y_truth = y_test[rand_num]
    total = np.zeros((64,13), dtype=float)
    # Create noise masks for ensemble
    for players in range(ensemble):
        noise = np.random.default_rng().normal(0.0, 0.1, size=(64,13))
        # Add mask to board and evaluate
        n_sample = np.add(x_sample, noise)
        n_predict = model(n_sample)
        n_predict = np.array(n_predict).reshape(1,64,13)
        # Sum outputs
        total = np.add(total, n_predict)
    # Get average of predictions
    av_predict = total / ensemble
    # Convert prediction categorical probabilities and truth one-hot array into strings of category labels
    pred_str = one_hot_to_unicode(av_predict)
    puzzle_str = one_hot_to_unicode(y_truth)
    # Compare strings
    if pred_str == puzzle_str:
        count += 1
    score = str(round(100*count/500, 2))
print(score, '% of puzzles accurately solved')