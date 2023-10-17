import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
import random
from _0_chess_tools import one_hot_to_unicode, any_cloned_pieces


# Get data from chess_encoder.py
print('loading pickled data...')
with open("mate_in_2.pickle", "rb") as file:
    mate_in_two = pickle.load(file) 
with open("solution.pickle", "rb") as file:
    solution = pickle.load(file) 

# Create datasets
x_test = np.array(mate_in_two[1300000:], dtype='bool')
y_test = np.array(solution[1300000:], dtype='bool')
print(x_test.shape, y_test.shape, '\n')

# Load models from chess_trainer.py
model_0 = keras.models.load_model('models/matein2_0')
model_1 = keras.models.load_model('models/matein2_1')
# model_2 = keras.models.load_model('models/matein2_2')
# model_3 = keras.models.load_model('models/matein2_3')
# model_4 = keras.models.load_model('models/matein2_4')
# model_5 = keras.models.load_model('models/matein2_5')
# model_6 = keras.models.load_model('models/matein2_6')
# model_7 = keras.models.load_model('models/matein2_7')
# model_8 = keras.models.load_model('models/matein2_8')
model_10 = keras.models.load_model('models/matein2')
models = [model_10, model_0, model_1]#, model_2, model_3, model_4, model_5, model_6, model_7, model_8]

# Calculate ensemble accuracy rate
count = 0
at_least_one = 0
rejected = 0
no_valid_preds = 0
for i in (range(10000)):
    # Get random board
    rand_num = random.sample(range(len(x_test)), 1)
    x_sample = x_test[rand_num]
    y_truth = y_test[rand_num]
    # Evaluate board with every model
    total = np.zeros((64,13), dtype=float)
    strike = 0
    valid_preds = 0
    for model in models:
        y_predict = model(x_sample)
        y_predict = np.array(y_predict).reshape(1,64,13)
        # Check if solo prediction is correct and record stats
        pred_str = one_hot_to_unicode(y_predict)
        puzzle_str = one_hot_to_unicode(y_truth)
        if pred_str == puzzle_str:
            strike += 1
        # Reject solo prediction if any piece count increases
        if any_cloned_pieces(x_sample, y_predict) == True:
            rejected += 1
        else:
            total = np.add(total, y_predict)
            valid_preds += 1
    # Find average of remaining solo predictions
    if valid_preds == 0:
        no_valid_preds += 1
        av_predict = y_predict
    else:
        av_predict = total / valid_preds
    # Convert prediction categorical probabilities and truth one-hot array into strings of category labels
    pred_str = one_hot_to_unicode(av_predict)
    puzzle_str = one_hot_to_unicode(y_truth)
    # Compare strings and tally results
    if pred_str == puzzle_str:
        count += 1
    if strike != 0:
        at_least_one += 1
    if i % 1000 == 0:
        print(i, 'solved')
score = str(round(100*count/10000, 2))

print(score, '% of puzzles accurately solved by ensemble average')
print(100*at_least_one/10000, '% of puzzles accurately solved by at least one model')
print(100*rejected/30000, '% non-sensible solo solves rejected')
print(100*no_valid_preds/10000, '% of puzzles with no valid solo predictions')


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
#mate in two:  (100% training data each), no cloned pieces (sometimes)
# 3. 53.5, 59.4, 16.4, 2.8
# 5. 56, 64.8 at least one, 16.6 rejected, 0.9 no valid solo
# 10. 57.5, 69.9, 16, 0.1
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