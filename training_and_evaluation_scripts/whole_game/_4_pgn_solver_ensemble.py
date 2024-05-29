""" Several neural networks are given the same chess board.  The solutions are combined according to
the decision criteria to produce an output that is substantially more accurate than any single neural 
network is able to produce. """

import numpy as np
import pickle
from tensorflow import keras
import random
from training_and_evaluation_scripts._0_chess_tools import *

# Get pickled data from encoder.py
with open("training data/all_puzz_w.pickle", "rb") as file:
    all_puzzles_1 = pickle.load(file) 

with open("training data/all_solns_w.pickle", "rb") as file:
    solutions_1 = pickle.load(file) 

# Create datasets
print('\ncreating test dataset...')
x_test = np.array(all_puzzles_1[1300000:], dtype='bool')
y_test = np.array(solutions_1[1300000:], dtype='bool')
# x_test = np.load('whole_game_x_test.npy')
# y_test = np.load('whole_game_y_test.npy')
print(x_test.shape, y_test.shape, '\n')

# Load models from chess_trainer.py
model_1 = keras.models.load_model('models/general_solver_1')
model_2 = keras.models.load_model('models/general_solver_2')
# model_3 = keras.models.load_model('models/general_solver_3')
# model_4 = keras.models.load_model('models/general_solver_4')
# model_5 = keras.models.load_model('models/whole_game_3')
# model_6 = keras.models.load_model('models/whole_game_4')
ensemble = [model_1, model_2]#, model_3, model_4]

# Calculate ensemble accuracy rate
remove_illegal = True
count = 0
at_least_one = 0
no_valid_preds = 0

for i in (range(3000)):
    # Get random board
    rand_num = random.sample(range(len(x_test)), 1)
    x_sample = x_test[rand_num]
    y_truth = y_test[rand_num]
    fen = one_hot_to_fen(x_sample, turn='white')
    allowed_moves = find_legal_moves(fen)
    
    # Reset variables
    raw_total = np.zeros((64,13), dtype=float)
    legal_total = np.zeros((64,13), dtype=float)
    max_lc_score =  -10000000000
    strike = 0
    legal_count = 0

    # Evaluate board with every model
    for model in ensemble:
        y_predict = model(x_sample)
        y_predict = np.array(y_predict).reshape(1,64,13)
        # Sum all predictions
        raw_total = np.add(raw_total, y_predict)
        # Check if solo output is correct and record stats
        pred_str = one_hot_to_unicode(y_predict)
        puzzle_str = one_hot_to_unicode(y_truth)
        if pred_str == puzzle_str:
            strike += 1

        # Compare prediction to all possible legal moves
        y_predict_bool = booleanise(y_predict)
        for tensor in allowed_moves:
            if np.all(y_predict_bool == tensor):
                legal_total = np.add(legal_total, y_predict)
                legal_count += 1

            # Get confidence score and save most confident legal prediction
            c_score = confid_score(y_predict) 
            if c_score > max_lc_score:
                mcf_leg_predict = y_predict
                max_lc_score = c_score
    
    if legal_count == 0:
        no_valid_preds += 1
    # Find average of all predictions and all legal predictions
    avg_raw_predict = raw_total / len(ensemble)
    avg_leg_predict = legal_total                  # no need to divide due to later argmax() 
    
    # Apply criteria to choose best prediction
    avg_raw_bool = booleanise(avg_raw_predict)
    for tensor in allowed_moves:
        if np.all(avg_raw_bool == tensor):
            ensemble_predict = avg_raw_predict

        else:
            avg_legal_bool = booleanise(avg_leg_predict)
            for tensor in allowed_moves:
                if np.all(avg_legal_bool == tensor):
                    ensemble_predict = avg_leg_predict

                else:
                    if legal_count >= 1:
                        # Use most confident legal solo prediction
                        ensemble_predict = mcf_leg_predict

                    else:
                        ensemble_predict = most_similar_move(allowed_moves, avg_raw_predict) # can this just be 'fen'?
                   
    # Convert prediction categorical probabilities and truth one-hot array into strings of category labels
    pred_str = one_hot_to_unicode(ensemble_predict)
    puzzle_str = one_hot_to_unicode(y_truth)
    # Compare strings and tally results
    if pred_str == puzzle_str:
        count += 1
    if strike != 0:
        at_least_one += 1
    if i % 1000 == 0:
        print(i, 'solved')

score = str(round(100*count/3000, 2))

print(score, '% of puzzles accurately solved by ensemble average')
print(100*at_least_one/3000, '% of puzzles accurately solved by at least one model')
print(100*no_valid_preds/3000, '% of puzzles with no valid solo predictions')



# whole_game_4 (+ whole_game_3)
# 31.3 (34.6)% of puzzles accurately solved by ensemble average
# 12.6 (16.1)% of puzzles accurately solved by at least one model
# 76.2 (77.4)% non-sensible solo solves rejected
# 5.4 (5.2)% illegal solo solves rejected
# 76.2 (68.1)% of puzzles with no valid solo predictions
# 28.1 (30.4)% of puzzles accurately solved using most_similar_legal_move() only

# mslm() update
# 1 model  last:  58.8%  first: 57.5%  
# 2 models  last: 60.1%  first: 59.4%
# 3 models  last: 61.4%  first: 60.7%
# 4 models  last: 62.8%  first: 61.6%

# mate in one: (25% of training data each)
# average solo: 66.2% win, 
# 4x ensemble: 72.2% win, at least one: 81.4%  ...try error rejection
# ...78.2% win, solo predictions rejected: 26.9%, no valid predictions: 8.3%  ...try illegal move rejection
# ...80.1% win, illegal moves rejected: 3.8%
# which curve fits best? rectangular hyperbola asymptote, vs power law improvement rate declines
#mate in two:  (100% of training data each)
# 1. 51% solved, 51% at least one
# 2. 47.1%, 57.6%
# 4. 44.1%, 63.3%
# 6. 44.7%, 65.8%
# 8. 47.8%, 68.4%
#10. 47.6%, 69.9%
#mate in one:  (25% training data)
# 1. 65.9% solved, 65.9% at least one
# 2. 67.9%, 75.7%
# 3. 66.2%, 79.0%
# 4. 66.0%, 81.8%
# 1. 65.6%, 65.6%, 27.7% rejected, 3.9% illegal, 27.7% no valid predictions
# 2. 74.3%, 75.2%, 27.3%, 3.6%, 16.3%
# 3. 78.5%, 79.5%, 27.1%, 3.7%, 10.7%
# 4. 79.6%, 81.5%, 27.2%, 3.9%, 7.8%
# 5. no change * (100% training data 78% acc.)
#mate in one:  (100% training data)
#1. 78.2, 78.8, 3.4, 0.5, 17
#2. 83.1, 84, 7.2, 1, 9.7
#3. 86, 87.2, 10.6, 1.4, 6.2
#4. 87.1, 88.7, 14.2, 1.9, 4.6

# 1 general solvers
# 32.18 % of puzzles accurately solved by ensemble average
# 29.23 % of puzzles accurately solved by at least one model
# 56.13 % non-sensible solo solves rejected
# 9.32 % illegal solo solves rejected
# 56.13 % of puzzles with no valid solo predictions

# 3 general solvers - focused_confid_score() 
# 44.37 % of puzzles accurately solved by ensemble average
# 42.34 % of puzzles accurately solved by at least one model
# 57.24 % non-sensible solo solves rejected
# 9.21 % illegal solo solves rejected
# 30.05 % of puzzles with no valid solo predictions

# 1000 puzzles solved in 110 seconds with 4 solvers

